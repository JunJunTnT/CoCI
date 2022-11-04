import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg.trainer.maddpg_original import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer_W_copy

import itertools


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.compat.v1.group(*expression)
    return U.function([], [], updates=[expression])

gate = 0.0

def p_train(make_obs_ph_n, weigths_ph, p_W, act_space_n, before_com_func, channel, after_com_func,
            q_func, optimizer, capacity=2, grad_norm_clipping=None, dim_message=14, num_units=64, scope="trainer",
            reuse=None, beta=0.01):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(num_agents)]

        hiddens_n = [before_com_func(obs_ph_n[i], dim_message, scope="before_com_{}".format(i), num_units=num_units) for
                     i in range(num_agents)]
        before_com_vars_n = [U.scope_vars(U.absolute_scope_name("before_com_{}".format(i))) for i in range(num_agents)]

        # todo stop gradient from hiddens
        hiddens_n_for_message = tf.compat.v1.stack(
            [before_com_func(obs_ph_n[i], dim_message, scope="before_com_{}".format(i), reuse=True, num_units=num_units)
             for i in range(num_agents)], axis=0)  # [num_agent, batch_size, dim_message]

        hiddens_n_for_message = tf.compat.v1.transpose(hiddens_n_for_message, [1, 0, 2])  # [batch_size, num_agent, dim_message]
        hiddens_n_for_message = tf.compat.v1.stop_gradient(hiddens_n_for_message)

        weigths = [p_W(obs_ph_n[i], 1, scope="weigths_{}".format(i), num_units=num_units) for i in range(num_agents)]
        weigths_vars = [U.scope_vars(U.absolute_scope_name("weigths_{}".format(i))) for i in range(num_agents)]
        p_n = []
        a_n = []
        num_com_agents = 2
        for i in range(num_agents):
            if i < num_com_agents:
                X_new = tf.compat.v1.greater_equal(weigths[i][0], gate)  # 0.5 as gate
                X_new = tf.compat.v1.cast(X_new, dtype=tf.compat.v1.float32)
                matrix_new = tf.compat.v1.tile(X_new, [dim_message])
                a_n.append(tf.compat.v1.multiply(hiddens_n[3], matrix_new))

                # p = after_com_func(obs_ph_n[i], int(act_pdtype_n[i].param_shape()[0]),
                #                    scope="p_func_{}".format(i), num_units=num_units)
                p = after_com_func(tf.compat.v1.concat([obs_ph_n[i], a_n[i]], 1), int(act_pdtype_n[i].param_shape()[0]),
                                   scope="p_func_{}".format(i), num_units=num_units)
            else:
                p = after_com_func(obs_ph_n[i], int(act_pdtype_n[i].param_shape()[0]),
                                   scope="p_func_{}".format(i), num_units=num_units)
            p_n.append(p)
        channel_vars_n = [U.scope_vars(U.absolute_scope_name("channel" + str(i))) for i in range(num_agents)]
        # todo check plus or concat the hidden and message
        #        p_n = [after_com_func(hiddens_n[i]+messages_list_batch[i], int(act_pdtype_n[i].param_shape()[0]), scope="p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]
        p_func_vars = [U.scope_vars(U.absolute_scope_name("p_func_{}".format(i))) for i in range(num_agents)]

        # wrap parameters in distribution
        act_pd_n = [act_pdtype_n[i].pdfromflat(p_n[i]) for i in range(num_agents)]

        act_sample_n = [act_pd.sample() for act_pd in act_pd_n]
        p_reg_n = [tf.compat.v1.reduce_mean(tf.compat.v1.square(act_pd.flatparam())) for act_pd in act_pd_n]

        act_input_n_n = [act_ph_n + [] for _ in range(num_agents)]
        weigths_input_n_n = [weigths_ph + [] for _ in range(num_agents)]
        for i in range(num_agents):
            act_input_n_n[i][i] = act_pd_n[i].sample()
            weigths_input_n_n[i][i] = weigths[i]
        q_input_n = [tf.compat.v1.concat(obs_ph_n + act_input_n_n[i], 1) for i in range(num_agents)]
        q_n = [q_func(q_input_n[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0] for i in
               range(num_agents)]
        pg_loss_n = [-tf.compat.v1.reduce_mean(q) for q in q_n]

        pg_loss = tf.compat.v1.reduce_sum(pg_loss_n)
        p_reg = tf.compat.v1.reduce_sum(p_reg_n)
        loss = pg_loss + p_reg * 1e-3

        w_q_input_n = [tf.compat.v1.concat(obs_ph_n + weigths_input_n_n[i], 1) for i in range(num_agents)]
        w_q_n = [q_func(w_q_input_n[i], 1, scope="w_q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0] for i
                 in range(num_agents)]
        w_pg_loss_n = [-tf.compat.v1.reduce_mean(q) for q in w_q_n]
        w_pg_loss = tf.compat.v1.reduce_sum(w_pg_loss_n)
        w_loss = w_pg_loss

        var_list = []
        var_list.extend(before_com_vars_n)
        # var_list.extend(channel_vars_n)
        var_list.extend(p_func_vars)
        #        var_list.extend(weigths_vars)
        var_list = list(itertools.chain(*var_list))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)
        optimize_expr_w = U.minimize_and_clip(optimizer, w_loss, weigths_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        train_w = U.function(inputs=obs_ph_n + weigths_ph, outputs=w_loss, updates=[optimize_expr_w])

        act = U.function(inputs=obs_ph_n, outputs=[act_sample_n, weigths, hiddens_n[3]])
        p_values = U.function(inputs=obs_ph_n, outputs=p_n)

        # target network
        target_hiddens_n = [
            before_com_func(obs_ph_n[i], dim_message, scope="target_before_com_{}".format(i), num_units=num_units) for i
            in range(num_agents)]
        target_before_com_vars = [U.scope_vars(U.absolute_scope_name("target_before_com_{}".format(i))) for i in
                                  range(num_agents)]

        target_hiddens_n_for_message = tf.compat.v1.stack([before_com_func(obs_ph_n[i], dim_message,
                                                                 scope="target_before_com_{}".format(i), reuse=True,
                                                                 num_units=num_units) for i in range(num_agents)],
                                                axis=0)  # [num_agent, batch_size, num_unints]
        target_hiddens_n_for_message = tf.compat.v1.transpose(target_hiddens_n_for_message,
                                                    [1, 0, 2])  # [batch_size, num_agent, num_unints]
        target_hiddens_n_for_message = tf.compat.v1.stop_gradient(target_hiddens_n_for_message)

        target_weigths = [p_W(obs_ph_n[i], 1, scope="target_weigths_{}".format(i), num_units=num_units) for i in
                          range(num_agents)]
        target_weigths_vars = [U.scope_vars(U.absolute_scope_name("target_weigths_{}".format(i))) for i in
                               range(num_agents)]

        target_p_n = []
        target_a_n = []
        for i in range(num_agents):
            if i < num_com_agents:

                X_new_ = tf.compat.v1.greater_equal(target_weigths[i][0], gate)  # 0.5 as gate
                X_new_ = tf.compat.v1.cast(X_new_, dtype=tf.compat.v1.float32)
                matrix_new_ = tf.compat.v1.tile(X_new_, [dim_message])
                target_a_n.append(tf.compat.v1.multiply(target_hiddens_n[3], matrix_new_))

                # p_ = after_com_func(obs_ph_n[i],
                #                     int(act_pdtype_n[i].param_shape()[0]),
                #                     scope="target_p_func_{}".format(i), num_units=num_units)


                p_ = after_com_func(tf.compat.v1.concat([obs_ph_n[i], target_a_n[i]], 1), int(act_pdtype_n[i].param_shape()[0]),
                                    scope="target_p_func_{}".format(i), num_units=num_units)
            else:
                p_ = after_com_func(obs_ph_n[i],
                                    int(act_pdtype_n[i].param_shape()[0]),
                                    scope="target_p_func_{}".format(i), num_units=num_units)

            target_p_n.append(p_)

        target_channel_vars_n = [U.scope_vars(U.absolute_scope_name("target_channel" + str(i))) for i in
                                 range(num_agents)]

        #        target_p_n = [after_com_func(target_hiddens_n[i]+target_message_n[i], int(act_pdtype_n[i].param_shape()[0]), scope="target_p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]
        target_p_func_vars = [U.scope_vars(U.absolute_scope_name("target_p_func_{}".format(i))) for i in
                              range(num_agents)]

        target_var_list = []
        target_var_list.extend(target_before_com_vars)
        # target_var_list.extend(target_channel_vars_n)
        target_var_list.extend(target_p_func_vars)
        #        target_var_list.extend(target_weigths_vars)
        target_var_list = list(itertools.chain(*target_var_list))
        update_target_p = make_update_exp(var_list, target_var_list)
        update_target_p_w = make_update_exp(list(itertools.chain(*weigths_vars)),
                                            list(itertools.chain(*target_weigths_vars)))

        target_act_sample_n = [act_pdtype_n[i].pdfromflat(target_p_n[i]).sample() for i in range(num_agents)]
        target_act = U.function(inputs=obs_ph_n, outputs=[target_act_sample_n, target_weigths])

        # check_message_n = U.function(inputs=obs_ph_n, outputs=message_n)
        check_hiddens_n = U.function(inputs=obs_ph_n, outputs=hiddens_n)

        return act, train, train_w, update_target_p, update_target_p_w, {'p_values': p_values, 'target_act': target_act,
                                                                         # 'check_message_n': check_message_n,
                                                                         'check_hiddens_n': check_hiddens_n}


def q_train(make_obs_ph_n, act_space_n, q_func, optimizer, grad_norm_clipping=None, scope="trainer",
            reuse=None, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_{}".format(i)) for i in
                    range(len(act_space_n))]
        target_ph_n = [tf.compat.v1.placeholder(tf.compat.v1.float32, [None], name="target_{}".format(i)) for i in range(num_agents)]

        q_input = tf.compat.v1.concat(obs_ph_n + act_ph_n, 1)
        q_n = [q_func(q_input, 1, scope="q_func_{}".format(i), num_units=num_units)[:, 0] for i in range(num_agents)]
        q_func_vars = [U.scope_vars(U.absolute_scope_name("q_func_{}".format(i))) for i in range(num_agents)]

        q_loss_n = [tf.compat.v1.reduce_mean(tf.compat.v1.square(q - target_ph)) for q, target_ph in zip(q_n, target_ph_n)]

        # viscosity solution to Bellman differential equation in place of an initial condition
        # q_reg = tf.compat.v1.reduce_mean(tf.compat.v1.square(q))
        q_loss = tf.compat.v1.reduce_sum(q_loss_n)
        loss = q_loss  # + 1e-3 * q_reg

        var_list = list(itertools.chain(*q_func_vars))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + target_ph_n, outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q_n)

        # target network
        target_q_n = [q_func(q_input, 1, scope="target_q_func_{}".format(i), num_units=num_units)[:, 0] for i in
                      range(num_agents)]
        target_q_func_vars = [U.scope_vars(U.absolute_scope_name("target_q_func_{}".format(i))) for i in
                              range(num_agents)]

        traget_var_list = list(itertools.chain(*target_q_func_vars))
        update_target_q = make_update_exp(var_list, traget_var_list)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q_n)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


def q_w_train(make_obs_ph_n, weigths, q_func, optimizer, grad_norm_clipping=None, scope="trainer",
              reuse=None, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        #        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_{}".format(i)) for i in range(len(act_space_n))]
        target_ph_n = [tf.compat.v1.placeholder(tf.compat.v1.float32, [None], name="w_target_{}".format(i)) for i in range(num_agents)]

        q_input = tf.compat.v1.concat(obs_ph_n + weigths, 1)
        q_n = [q_func(q_input, 1, scope="w_q_func_{}".format(i), num_units=num_units)[:, 0] for i in range(num_agents)]
        q_func_vars = [U.scope_vars(U.absolute_scope_name("w_q_func_{}".format(i))) for i in range(num_agents)]

        q_loss_n = [tf.compat.v1.reduce_mean(tf.compat.v1.square(q - target_ph)) for q, target_ph in zip(q_n, target_ph_n)]

        # viscosity solution to Bellman differential equation in place of an initial condition
        # q_reg = tf.compat.v1.reduce_mean(tf.compat.v1.square(q))
        q_loss = tf.compat.v1.reduce_sum(q_loss_n)
        loss = q_loss  # + 1e-3 * q_reg

        var_list = list(itertools.chain(*q_func_vars))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + weigths + target_ph_n, outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + weigths, q_n)

        # target network
        target_q_n = [q_func(q_input, 1, scope="w_target_q_func_{}".format(i), num_units=num_units)[:, 0] for i in
                      range(num_agents)]
        target_q_func_vars = [U.scope_vars(U.absolute_scope_name("w_target_q_func_{}".format(i))) for i in
                              range(num_agents)]

        traget_var_list = list(itertools.chain(*target_q_func_vars))
        update_target_q = make_update_exp(var_list, traget_var_list)

        target_q_values = U.function(obs_ph_n + weigths, target_q_n)

        return train, update_target_q, {'w_q_values': q_values, 'w_target_q_values': target_q_values}


class ACML_CTrainer(AgentTrainer):
    def __init__(self, name, before_com_model, channel, after_com_model, critic_mlp_model, model_weights,
                 obs_shape_n, act_space_n, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args
        self.update_interval = args.update_interval

        obs_ph_n = []
        #        network_obs = []
        weigths = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation_" + str(i)).get())
            #            network_obs.append(U.BatchInput(obs_shape_net[i], name="observation_net"+str(i)).get())
            weigths.append(U.BatchInput((1,), name="weigths" + str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_func=critic_mlp_model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units
        )
        self.w_q_train, self.w_q_update, self.w_q_debug = q_w_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            weigths=weigths,
            q_func=critic_mlp_model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units
        )
        self.act, self.p_train, self.w_train, self.p_update, self.w_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            #            network_input=network_obs,
            weigths_ph=weigths,
            p_W=model_weights,
            act_space_n=act_space_n,
            before_com_func=before_com_model,
            channel=channel,
            after_com_func=after_com_model,
            q_func=critic_mlp_model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            # beta=args.beta,
            # dim_message=args.dim_message,
            # capacity=args.capacity
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer_W_copy(1e6)
        # self.max_replay_buffer_len = 50 * args.max_episode_len
        self.max_replay_buffer_len = args.batch_size * 2
        self.replay_sample_index = None

    def action(self, obs_n):
        obs = [obs[None] for obs in obs_n]
        action_list = self.act(*(obs))
        # print(action_list)

        return [act[0].tolist() for act in action_list[0]], [act[0][0] for act in action_list[1]]

    def experience(self, obs, act, weight, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, weight, rew, new_obs, [float(d) for d in done])

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        # if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
        #     return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents

        index = self.replay_sample_index
        samples = self.replay_buffer.sample_index(index)
        obs_n, act_n, weigths_n, rew_n, obs_next_n, done_n = [np.swapaxes(item, 0, 1) for item in samples]
        # rew_com_n = rew_n - 0.05 * weigths_n
        # train q network

        [target_act_next_n, target_w_next_n] = self.p_debug['target_act'](*(obs_next_n.tolist()))

        target_q_next_n = self.q_debug['target_q_values'](*(obs_next_n.tolist() + target_act_next_n))
        target_q_n = [rew + self.args.gamma * (1.0 - done) * target_q_next for rew, done, target_q_next in
                      zip(rew_n, done_n, target_q_next_n)]
        target_q_n = [target_q for target_q in target_q_n]
        q_loss = self.q_train(*(obs_n.tolist() + act_n.tolist() + target_q_n))

        target_q_next_n_w = self.w_q_debug['w_target_q_values'](*(obs_next_n.tolist() + target_w_next_n))
        target_q_n_w = [rew + self.args.gamma * (1.0 - done) * target_q_next for rew, done, target_q_next in
                        zip(rew_n, done_n, target_q_next_n_w)]
        target_q_n_w = [target_q for target_q in target_q_n_w]

        # train p network
        p_loss = self.p_train(*(obs_n.tolist() + act_n.tolist()))

        self.p_update()
        self.q_update()

        w_q_loss = self.w_q_train(*(obs_n.tolist() + weigths_n.tolist() + target_q_n_w))
        w_p_loss = self.w_train(*(obs_n.tolist() + weigths_n.tolist()))
        self.w_update()
        self.w_q_update()

        return [q_loss, p_loss]

