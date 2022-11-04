import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer_T

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


def p_train(make_obs_ph_n, network_input, time_ph_n, time_space_n, p_timer, act_space_n, before_com_func, channel,
            after_com_func,
            q_func, optimizer, grad_norm_clipping=None, dim_message=8, num_units=64, scope="trainer", reuse=None,
            beta=0.01):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        timer_w_pdtype_n = [make_pdtype(time_w_space) for time_w_space in time_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(num_agents)]
        time_w_ph_n = [timer_w_pdtype_n[i].sample_placeholder([None], name="time_w" + str(i)) for i in
                       range(num_agents)]

        hiddens_n = [
            before_com_func(tf.compat.v1.concat([obs_ph_n[i], network_input[i]], 1), dim_message, scope="before_com_{}".format(i),
                            num_units=num_units) for i in range(num_agents)]
        before_com_vars_n = [U.scope_vars(U.absolute_scope_name("before_com_{}".format(i))) for i in range(num_agents)]

        # todo stop gradient from hiddens
        hiddens_n_for_message = tf.compat.v1.stack([hiddens_n[i] for i in range(num_agents)], axis=0)
        # tf.compat.v1.stack([before_com_func(tf.compat.v1.concat([obs_ph_n[i], network_input[i]], 1), dim_message, scope="before_com_{}".format(i), reuse=True, num_units=num_units)
        # for i in range(num_agents)], axis=0) # [num_agent, batch_size, dim_message]

        hiddens_n_for_message = tf.compat.v1.transpose(hiddens_n_for_message, [1, 0, 2])  # [batch_size, num_agent, dim_message]
        # hiddens_n_for_message = tf.compat.v1.stop_gradient(hiddens_n_for_message)

        #        timer_n = [p_timer(tf.compat.v1.concat([obs_ph_n[i], network_input[i]], 1), 1, scope="timer_{}".format(i), num_units=int(num_units/2)) for i in range(num_agents)]
        timer_generate_n = [
            p_timer(tf.compat.v1.concat([obs_ph_n[i], network_input[i]], 1), num_agents, scope="timer_{}".format(i),
                    num_units=num_units) for i in range(num_agents)]
        time_pd_n = [timer_w_pdtype_n[i].pdfromflat(timer_generate_n[i]) for i in range(num_agents)]
        timer_sample_n = [time_pd.sample() for time_pd in time_pd_n]

        max_indices = [tf.compat.v1.reshape(tf.compat.v1.argmax(timer_sample_n[i], axis=1), [-1, 1]) for i in range(num_agents)]
        x = tf.compat.v1.shape(network_input[0])[0]
        ii = tf.compat.v1.range(x)[:, tf.compat.v1.newaxis]
        ii = tf.compat.v1.cast(ii, dtype=tf.compat.v1.int64)
        idx = [tf.compat.v1.concat([ii, max_indices[i]], axis=1) for i in range(num_agents)]
        timer_n = [tf.compat.v1.reshape(tf.compat.v1.gather_nd(network_input[i], idx[i]), [-1, 1]) for i in range(num_agents)]
        timer_n = [timer_n[i] + 0.001 * tf.compat.v1.ones_like(timer_n[i]) for i in range(num_agents)]

        timer_vars = [U.scope_vars(U.absolute_scope_name("timer_{}".format(i))) for i in range(num_agents)]
        #        get_timer= U.function(inputs= [obs_ph_n[p_index]]+ [network_input[p_index]], outputs=timer)
        # timer from NN
        p_n = []
        for i in range(num_agents):
            X = tf.compat.v1.math.less_equal(network_input[i], timer_n[i])
            X = tf.compat.v1.cast(X, dtype=tf.compat.v1.float32)
            Z = tf.compat.v1.expand_dims(X, 2)
            time_matrix = tf.compat.v1.tile(Z, (1, 1, dim_message))
            messages_list_batch = tf.compat.v1.multiply(hiddens_n_for_message,
                                              time_matrix)  # [num_agent, batch_size, num_agent, mess_dim]

            channel_output = channel(messages_list_batch, scope="channel" + str(i),
                                     num_heads=2)  # [batch_size, num_agent, num_unints]
            channel_output = tf.compat.v1.transpose(channel_output, [1, 0, 2])  # [num_agent, batch_size, num_unints]
            # print(channel_output)
            message_n = [tf.compat.v1.squeeze(message, axis=0) for message in
                         tf.compat.v1.split(channel_output, num_or_size_splits=num_agents, axis=0)]

            p = after_com_func(tf.compat.v1.concat([hiddens_n[i], message_n[i]], 1), int(act_pdtype_n[i].param_shape()[0]),
                               scope="p_func_{}".format(i), num_units=num_units)
            p_n.append(p)

        #        # timer = 0
        #        p_n_0 = []
        #        for i in range(num_agents):
        #            message_n =hiddens_n
        #            p_0 = after_com_func(tf.compat.v1.concat([hiddens_n[i], message_n[i]],1), int(act_pdtype_n[i].param_shape()[0]), scope="p_func_{}".format(i), num_units=num_units)
        #            p_n_0.append(p_0)

        # timer from memory
        p_n_memeory = []
        for i in range(num_agents):
            X = tf.compat.v1.math.less_equal(network_input[i], time_ph_n[i])
            X = tf.compat.v1.cast(X, dtype=tf.compat.v1.float32)
            Z = tf.compat.v1.expand_dims(X, 2)
            time_matrix = tf.compat.v1.tile(Z, (1, 1, dim_message))
            messages_list_batch_ = tf.compat.v1.multiply(hiddens_n_for_message,
                                               time_matrix)  # [num_agent, batch_size, num_agent, mess_dim]

            channel_output_ = channel(messages_list_batch_, scope="channel" + str(i), num_heads=2,
                                      reuse=True)  # [batch_size, num_agent, num_unints]
            channel_output_ = tf.compat.v1.transpose(channel_output_, [1, 0, 2])  # [num_agent, batch_size, num_unints]
            # channel_output_ = tf.compat.v1.stop_gradient(channel_output_) #???????????
            # print(channel_output_)
            message_n_ = [tf.compat.v1.squeeze(message, axis=0) for message in
                          tf.compat.v1.split(channel_output_, num_or_size_splits=num_agents, axis=0)]

            p2 = after_com_func(tf.compat.v1.concat([hiddens_n[i], message_n_[i]], 1), int(act_pdtype_n[i].param_shape()[0]),
                                scope="p_func_{}".format(i), num_units=num_units, reuse=True)
            p_n_memeory.append(p2)

        channel_vars_n = [U.scope_vars(U.absolute_scope_name("channel" + str(i))) for i in range(num_agents)]
        p_func_vars = [U.scope_vars(U.absolute_scope_name("p_func_{}".format(i))) for i in range(num_agents)]

        # wrap parameters in distribution
        act_pd_n = [act_pdtype_n[i].pdfromflat(p_n_memeory[i]) for i in range(num_agents)]  # with memorized timer
        act_pd_n_t = [act_pdtype_n[i].pdfromflat(p_n[i]) for i in range(num_agents)]
        #        act_pd_n_t_0 = [act_pdtype_n[i].pdfromflat(p_n_0[i]) for i in range(num_agents)]

        act_sample_n = [act_pd.sample() for act_pd in act_pd_n_t]
        # act_sample_n_0 = [act_pd.sample() for act_pd in act_pd_n_t_0]
        p_reg_n = [tf.compat.v1.reduce_mean(tf.compat.v1.square(act_pd.flatparam())) for act_pd in act_pd_n]

        #        act_input_n_n = [act_ph_n + [] for _ in range(num_agents)]
        timer_input_n_n = [time_w_ph_n + [] for _ in range(num_agents)]
        act_input_n_n_actor = [act_ph_n + [] for _ in range(num_agents)]
        act_input_n_n_timer = [act_ph_n + [] for _ in range(num_agents)]
        #        act_input_n_n_timer_0 =[act_ph_n + [] for _ in range(num_agents)]
        timer_input_n_n_timer = [time_w_ph_n + [] for _ in range(num_agents)]
        timer_input_n_n_timer_0 = [time_w_ph_n + [] for _ in range(num_agents)]
        for i in range(num_agents):
            act_input_n_n_actor[i][i] = act_pd_n[i].sample()
            act_input_n_n_timer[i][i] = act_pd_n_t[i].sample()
            #            act_input_n_n_timer_0[i][i] = act_pd_n_t_0[i].sample()

            timer_input_n_n_timer[i][i] = timer_sample_n[i]

            #            timer_n[i]
            # print(timer_input_n_n_timer_0[i][i] )
            timer_input_n_n_timer_0[i][i] = tf.compat.v1.zeros_like(timer_input_n_n_timer_0[i][i], dtype=tf.compat.v1.float32)
        #            timer_input_n_n_timer_0[i][i][:,i]
        # tf.compat.v1.transpose(tf.compat.v1.zeros((1,)),[1,0])

        # Training actor => time from memory
        q_input_n_actor = [tf.compat.v1.concat(obs_ph_n + network_input + act_input_n_n_actor[i] + timer_input_n_n[i], 1) for i in
                           range(num_agents)]
        q_n_actor = [q_func(q_input_n_actor[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0]
                     for i in range(num_agents)]
        pg_loss_n_actor = [-tf.compat.v1.reduce_mean(q) for q in q_n_actor]
        pg_loss_actor = tf.compat.v1.reduce_sum(pg_loss_n_actor)
        p_reg = tf.compat.v1.reduce_sum(p_reg_n)
        loss_actor = pg_loss_actor + p_reg * 1e-3

        # Training timer => actions all from memory
        q_input_n_timer = [tf.compat.v1.concat(obs_ph_n + network_input + act_input_n_n_timer[i] + timer_input_n_n_timer[i], 1)
                           for i in range(num_agents)]
        # for i in range(4):
        #     print(q_input_n_timer[i].shape)
        # print("===")
        #        q_input_n_timer_0 = [tf.compat.v1.concat(obs_ph_n + network_input + act_input_n_n_timer_0[i] + timer_input_n_n_timer_0[i], 1) for i in range(num_agents)]
        q_n_t = [q_func(q_input_n_timer[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0] for i
                 in range(num_agents)]
        #        q_n_0 = [q_func(q_input_n_timer_0[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:,0] for i in range(num_agents)]
        pg_loss_n_timer = [-tf.compat.v1.reduce_mean(qt) for qt in q_n_t]
        #        [-tf.compat.v1.reduce_mean(qt-q0) for qt,q0 in zip( q_n_t, q_n_0)]
        pg_loss_timer = tf.compat.v1.reduce_sum(pg_loss_n_timer)
        loss_timer = pg_loss_timer

        var_list = []
        var_list.extend(before_com_vars_n)
        var_list.extend(channel_vars_n)
        var_list.extend(p_func_vars)
        # var_list.extend(timer_vars)
        var_list = list(itertools.chain(*var_list))
        optimize_expr = U.minimize_and_clip(optimizer, loss_actor, var_list, grad_norm_clipping)
        optimize_expr_timer = U.minimize_and_clip(optimizer, loss_timer, timer_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + network_input + time_ph_n + time_w_ph_n, outputs=loss_actor,
                           updates=[optimize_expr])
        train_timer = U.function(inputs=obs_ph_n + act_ph_n + network_input + time_ph_n + time_w_ph_n,
                                 outputs=loss_timer, updates=[optimize_expr_timer])
        act = U.function(inputs=obs_ph_n + network_input, outputs=[act_sample_n, timer_n, timer_sample_n])
        p_values = U.function(inputs=obs_ph_n + network_input, outputs=p_n)

        # target network
        target_hiddens_n = [before_com_func(tf.compat.v1.concat([obs_ph_n[i], network_input[i]], 1), dim_message,
                                            scope="target_before_com_{}".format(i), num_units=num_units) for i in
                            range(num_agents)]
        target_before_com_vars = [U.scope_vars(U.absolute_scope_name("target_before_com_{}".format(i))) for i in
                                  range(num_agents)]

        target_hiddens_n_for_message = tf.compat.v1.stack(
            [before_com_func(tf.compat.v1.concat([obs_ph_n[i], network_input[i]], 1), dim_message,
                             scope="target_before_com_{}".format(i), reuse=True,
                             num_units=num_units) for i in range(num_agents)],
            axis=0)  # [num_agent, batch_size, num_unints]
        target_hiddens_n_for_message = tf.compat.v1.transpose(target_hiddens_n_for_message,
                                                    [1, 0, 2])  # [batch_size, num_agent, num_unints]
        target_hiddens_n_for_message = tf.compat.v1.stop_gradient(target_hiddens_n_for_message)

        #        target_timer_generate_n = [p_timer(tf.compat.v1.concat([obs_ph_n[i],network_input[i]], 1), num_agents, scope="target_timer_{}".format(i),
        #                                num_units=num_units) for i in range(num_agents)]

        T_timer_generate_n = [
            p_timer(tf.compat.v1.concat([obs_ph_n[i], network_input[i]], 1), num_agents, scope="target_timer_{}".format(i),
                    num_units=num_units) for i in range(num_agents)]
        T_time_pd_n = [timer_w_pdtype_n[i].pdfromflat(T_timer_generate_n[i]) for i in range(num_agents)]
        T_timer_sample_n = [time_pd.sample() for time_pd in T_time_pd_n]

        max_indices_ = [tf.compat.v1.reshape(tf.compat.v1.argmax(T_timer_sample_n[i], axis=1), [-1, 1]) for i in range(num_agents)]
        x_ = tf.compat.v1.shape(network_input[0])[0]
        ii_ = tf.compat.v1.range(x_)[:, tf.compat.v1.newaxis]
        ii_ = tf.compat.v1.cast(ii_, dtype=tf.compat.v1.int64)
        idx_ = [tf.compat.v1.concat([ii_, max_indices_[i]], axis=1) for i in range(num_agents)]
        target_timer_n = [tf.compat.v1.reshape(tf.compat.v1.gather_nd(network_input[i], idx_[i]), [-1, 1]) for i in range(num_agents)]
        target_timer_n = [target_timer_n[i] + 0.001 * tf.compat.v1.ones_like(target_timer_n[i]) for i in range(num_agents)]

        target_timer_vars = [U.scope_vars(U.absolute_scope_name("target_timer_{}".format(i))) for i in
                             range(num_agents)]
        target_p_n = []
        for i in range(num_agents):
            X_ = tf.compat.v1.math.less_equal(network_input[i], target_timer_n[i])
            X_ = tf.compat.v1.cast(X_, dtype=tf.compat.v1.float32)
            Z_ = tf.compat.v1.expand_dims(X_, 2)
            target_time_matrix = tf.compat.v1.tile(Z_, (1, 1, dim_message))
            target_messages_list_batch = tf.compat.v1.multiply(target_hiddens_n_for_message,
                                                     target_time_matrix)  # [num_agent, batch_size, num_agent, mess_dim]
            target_channel_output = channel(target_messages_list_batch, scope="target_channel" + str(i),
                                            num_heads=2)  # [batch_size, num_agent, num_unints]
            target_channel_output = tf.compat.v1.transpose(target_channel_output,
                                                 [1, 0, 2])  # [num_agent, batch_size, num_unints]
            target_message_n = [tf.compat.v1.squeeze(message, axis=0) for message in
                                tf.compat.v1.split(target_channel_output, num_or_size_splits=num_agents, axis=0)]
            p_ = after_com_func(tf.compat.v1.concat([target_hiddens_n[i], target_message_n[i]], 1),
                                int(act_pdtype_n[i].param_shape()[0]), scope="target_p_func_{}".format(i),
                                num_units=num_units)
            #            target_hiddens_n[i]+target_message_n[i] + target_timer_n[i]
            target_p_n.append(p_)
        target_channel_vars_n = [U.scope_vars(U.absolute_scope_name("target_channel" + str(i))) for i in
                                 range(num_agents)]
        #        target_p_n = [after_com_func(target_hiddens_n[i]+target_message_n[i], int(act_pdtype_n[i].param_shape()[0]), scope="target_p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]
        target_p_func_vars = [U.scope_vars(U.absolute_scope_name("target_p_func_{}".format(i))) for i in
                              range(num_agents)]

        target_var_list = []
        target_var_list.extend(target_before_com_vars)
        target_var_list.extend(target_channel_vars_n)
        target_var_list.extend(target_p_func_vars)
        # target_var_list.extend(target_timer_vars)
        target_var_list = list(itertools.chain(*target_var_list))
        # print(var_list)
        # print(target_channel_vars_n)
        # print(timer_vars)

        update_target_p = make_update_exp(var_list, target_var_list)
        update_target_timer = make_update_exp(list(itertools.chain(*timer_vars)),
                                              list(itertools.chain(*target_timer_vars)))

        target_act_sample_n = [act_pdtype_n[i].pdfromflat(target_p_n[i]).sample() for i in range(num_agents)]
        target_act = U.function(inputs=obs_ph_n + network_input, outputs=[target_act_sample_n, T_timer_sample_n])



        check_message_n = U.function(inputs=obs_ph_n + network_input, outputs=message_n)
        check_hiddens_n = U.function(inputs=obs_ph_n, outputs=hiddens_n)

        return act, train, train_timer, update_target_p, update_target_timer, {'p_values': p_values,
                                                                               'target_act': target_act,
                                                                               'check_message_n': check_message_n,
                                                                               'check_hiddens_n': check_hiddens_n}


def q_train(make_obs_ph_n, network_input, time_space_n, act_space_n, q_func, optimizer, grad_norm_clipping=None,
            scope="trainer",
            reuse=None, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        time_pdtype_n = [make_pdtype(t_space) for t_space in time_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_{}".format(i)) for i in
                    range(len(act_space_n))]
        time_ph_n_w = [time_pdtype_n[i].sample_placeholder([None], name="time_w_{}".format(i)) for i in
                       range(len(time_space_n))]

        target_ph_n = [tf.compat.v1.placeholder(tf.compat.v1.float32, [None], name="target_{}".format(i)) for i in range(num_agents)]

        q_input = tf.compat.v1.concat(obs_ph_n + network_input + act_ph_n + time_ph_n_w, 1)
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
        train = U.function(inputs=obs_ph_n + network_input + act_ph_n + time_ph_n_w + target_ph_n, outputs=loss,
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + network_input + act_ph_n + time_ph_n_w, q_n)

        # target network
        target_q_n = [q_func(q_input, 1, scope="target_q_func_{}".format(i), num_units=num_units)[:, 0] for i in
                      range(num_agents)]
        target_q_func_vars = [U.scope_vars(U.absolute_scope_name("target_q_func_{}".format(i))) for i in
                              range(num_agents)]

        traget_var_list = list(itertools.chain(*target_q_func_vars))
        update_target_q = make_update_exp(var_list, traget_var_list)

        target_q_values = U.function(obs_ph_n + network_input + act_ph_n + time_ph_n_w, target_q_n)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class TACAgentTrainer_PL(AgentTrainer):
    def __init__(self, name, before_com_model, channel, after_com_model, critic_mlp_model, model_timer,
                 obs_shape_n, obs_shape_net, act_space_n, timer_w_space_n, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args
        self.update_interval = args.update_interval

        obs_ph_n = []
        network_obs = []
        timer = []
        #        timer_w_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation_" + str(i)).get())
            network_obs.append(U.BatchInput(obs_shape_net[i], name="observation_net" + str(i)).get())
            timer.append(U.BatchInput((1,), name="timer_m" + str(i)).get())
        #            timer_w_n.append(U.BatchInput((self.n,), name="timer_w"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            network_input=network_obs,
            #            time_ph_n = timer,
            time_space_n=timer_w_space_n,
            #            time_ph_n_w = timer_w_space_n,
            act_space_n=act_space_n,
            q_func=critic_mlp_model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
        )
        self.act, self.p_train, self.timer_train, self.p_update, self.timer_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            network_input=network_obs,
            time_ph_n=timer,
            time_space_n=timer_w_space_n,
            p_timer=model_timer,
            act_space_n=act_space_n,
            before_com_func=before_com_model,
            channel=channel,
            after_com_func=after_com_model,
            q_func=critic_mlp_model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer_T(1e6)
        # self.max_replay_buffer_len = 50 * args.max_episode_len
        self.max_replay_buffer_len = args.batch_size * 2
        self.replay_sample_index = None

    #    def action(self, obs_n):
    #        obs = [obs[None] for obs in obs_n]
    #         
    #        return self.act(*obs)

    def action(self, obs_n, obsnet_n, Epi):
        # print(obs_n)
        # obsnet_n = [obsnet_n]
        # print(obsnet_n)
        obs = [obs[None] for obs in obs_n]
        # print(obs)
        obsnet = [obsnet[None] for obsnet in obsnet_n]
        action_list = self.act(*(obs + obsnet))
        return [act[0].tolist() for act in action_list[0]], [act[0][0] for act in action_list[1]], [act[0].tolist() for
                                                                                                    act in
                                                                                                    action_list[2]]

    #    def experience(self, obs, act, rew, new_obs, done):
    #        # Store transition in the replay buffer.
    #        self.replay_buffer.add(obs, act, rew, new_obs, [float(d) for d in done])

    def experience(self, obs, obsnet, time, time_w, act, rew, new_obs, new_obsnet, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, obsnet, time, time_w, act, rew, new_obs, new_obsnet, [float(d) for d in done])

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        #         if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
        #             return
            if not t % 100 == 0:  # only update every 100 steps
                return

            self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
            # collect replay sample from all agents
            obs_n = []
            obs_next_n = []
            act_n = []
            index = self.replay_sample_index
            samples = self.replay_buffer.sample_index(index)
            # for elem in samples:
            #     print(elem.shape)
            #     print(elem)
                # np.swapaxes(elem, 0, 1)
                # print("===")
            obs_n, obsnet_n, time_n, time_w_n, act_n, rew_n, obs_next_n, obsnet_next_n, done_n = [np.swapaxes(item, 0, 1) for item in samples]
                # train q network

        #        target_q = 0.0
        #     print(np.array(obs_next_n).shape)
        #     print(obs_next_n)
        #     print(obsnet_n)
        #     print(np.array(obsnet_next_n).shape)
            target_act_next_list = self.p_debug['target_act'](*(obs_next_n.tolist() + obsnet_next_n.tolist()))

            target_act_next_n, target_w_next_n = target_act_next_list[0], target_act_next_list[1]

            target_q_next_n = self.q_debug['target_q_values'](*(obs_next_n.tolist() + obsnet_next_n.tolist() + target_act_next_n + target_w_next_n))
            target_q_n = [rew + self.args.gamma * (1.0 - done) * target_q_next for rew, done, target_q_next in
                            zip(rew_n, done_n, target_q_next_n)]
            target_q_n = [target_q for target_q in target_q_n]
            q_loss = self.q_train(*(obs_n.tolist() + obsnet_n.tolist() + act_n.tolist() + time_w_n.tolist()+ target_q_n))

                # train p network
            p_loss = self.p_train(*(obs_n.tolist()+ act_n.tolist() + obsnet_n.tolist()  + time_n.tolist()+ time_w_n.tolist()))
            timer_loss = self.timer_train(*(obs_n.tolist()+ act_n.tolist() + obsnet_n.tolist()  + time_n.tolist()+ time_w_n.tolist()))

            self.p_update()
            self.timer_update()
            self.q_update()
            # if t % 5000 == 0:
            #     message_n = self.p_debug['check_message_n'](*(obs_n.tolist()))
            #     hiddens_n = self.p_debug['check_hiddens_n'](*(obs_n.tolist()))
            #     print("message_n", message_n[0][0])
            #     print("hiddens_n", hiddens_n[0][0])
        # q_loss, p_loss, timer_loss = 0, 0, 0
            return [q_loss, p_loss, timer_loss]

