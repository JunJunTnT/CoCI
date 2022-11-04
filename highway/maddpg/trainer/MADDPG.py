import itertools
import numpy as np
import tensorflow as tf
import maddpg.common.tf_util as U
from maddpg.common.distributions import make_pdtype
from maddpg.trainer.maddpg_original import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer_W

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def f1():
    return 0;


def f2():
    return 1;


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.compat.v1.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, weights_ph, p_W, act_space_n, p_func,
            q_func,  optimizer, gate, packet_loss_rate, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        weights = [p_W(obs_ph_n[i], 1, scope="weigths_{}".format(i), num_units=num_units) for i in range(num_agents)]
        weights_vars = [U.scope_vars(U.absolute_scope_name("weigths_{}".format(i))) for i in range(num_agents)]

        p_n = []
        new_obs_n = []
        for i in range(num_agents):
            # ----------------improve com-----------#
            X = tf.compat.v1.greater_equal(weights[i][0], gate)  # 0.5 as gate
            X = tf.compat.v1.cast(X, dtype=tf.compat.v1.float32)
            matrix = tf.compat.v1.tile(X, multiples=[5])
            matrix = tf.compat.v1.cast(matrix, dtype=tf.compat.v1.float32)
            Y = tf.compat.v1.tile([1], multiples=[5])
            Y = tf.compat.v1.cast(Y, dtype=tf.compat.v1.float32)
            Y2 = tf.compat.v1.tile([0], multiples=[5])
            Y2 = tf.compat.v1.cast(Y2, dtype=tf.compat.v1.float32)

            # paket loss testing #
            if np.random.random() < packet_loss_rate:
                matrix2 = tf.compat.v1.tile([0], multiples=[5])
            else:
                matrix2 = tf.compat.v1.tile([1], multiples=[5])
            matrix2 = tf.compat.v1.cast(matrix2, dtype=tf.compat.v1.float32)
            matrix = tf.compat.v1.multiply(matrix, matrix2)

            Z = tf.compat.v1.concat([Y, matrix, matrix, matrix, matrix], 0)
            Z2 = tf.compat.v1.concat([Y, Y, Y, matrix, matrix], 0)
            Z3 = tf.compat.v1.concat([Y, Y, Y, Y, Y], 0)

            if i < 2:
                new_obs = tf.compat.v1.multiply(make_obs_ph_n[i], Z)
            elif i == 2:
                new_obs = tf.compat.v1.multiply(make_obs_ph_n[i], Z2)
            else:
                new_obs = tf.compat.v1.multiply(make_obs_ph_n[i], Z3)

            new_obs_n.append(new_obs)
            p = p_func(new_obs, int(act_pdtype_n[i].param_shape()[0]), scope="p_func_{}".format(i), num_units=num_units)
            p_n.append(p)


        # p_n=[p_func(new_obs_n[i], int(act_pdtype_n[i].param_shape()[0]), scope="p_func_{}".format(i), num_units=num_units)
        #      for i in range(num_agents)]
        p_func_vars = [U.scope_vars(U.absolute_scope_name("p_func_{}".format(i))) for i in range(num_agents)]


        # wrap parameters in distribution
        act_pd_n = [act_pdtype_n[i].pdfromflat(p_n[i]) for i in range(num_agents)]

        act_sample_n = [act_pd.sample() for act_pd in act_pd_n]
        p_reg_n = [tf.compat.v1.reduce_mean(tf.compat.v1.square(act_pd.flatparam())) for act_pd in act_pd_n]

        act_input_n_n = [act_ph_n + [] for _ in range(num_agents)]
        weights_input_n_n = [weights_ph + [] for _ in range(num_agents)]

        for i in range(num_agents):
            act_input_n_n[i][i] = act_pd_n[i].sample()
            weights_input_n_n[i][i] = weights[i]

        q_input_n = [tf.compat.v1.concat(new_obs_n + act_input_n_n[i], 1) for i in range(num_agents)]

        q_n = [q_func(q_input_n[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0]
               for i in range(num_agents)]
        pg_loss_n = [-tf.compat.v1.reduce_mean(q) for q in q_n]

        pg_loss = tf.compat.v1.reduce_sum(pg_loss_n)
        p_reg = tf.compat.v1.reduce_sum(p_reg_n)
        loss = pg_loss + p_reg * 1e-3

        w_q_input_n = [tf.compat.v1.concat(obs_ph_n + weights_input_n_n[i], 1) for i in range(num_agents)]
        w_q_n = [q_func(w_q_input_n[i], 1, scope="w_q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0]
                 for i in range(num_agents)]
        w_pg_loss_n = [-tf.compat.v1.reduce_mean(q) for q in w_q_n]
        w_pg_loss = tf.compat.v1.reduce_sum(w_pg_loss_n)
        w_loss = w_pg_loss


        var_list = []
        var_list.extend(p_func_vars)
        var_list = list(itertools.chain(*var_list))

        w_var_list = []
        w_var_list.extend(weights_vars)
        w_var_list = list(itertools.chain(*w_var_list))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)
        optimize_expr_w = U.minimize_and_clip(optimizer, w_loss, w_var_list, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n , outputs=loss, updates=[optimize_expr])
        #train = U.function(inputs=obs_ph_n + act_ph_n + weights_ph, outputs=loss, updates=[optimize_expr])
        train_w = U.function(inputs=obs_ph_n + weights_ph, outputs=w_loss, updates=[optimize_expr_w])

        act = U.function(inputs=obs_ph_n, outputs=[act_sample_n, weights])
        p_values = U.function(obs_ph_n, p_n)

        # target network

        target_weights = [p_W(obs_ph_n[i], 1, scope="target_weigths_{}".format(i), num_units=num_units) for i in range(num_agents)]
        target_weights_vars = [U.scope_vars(U.absolute_scope_name("target_weigths_{}".format(i))) for i in range(num_agents)]


        target_p_n = []
        for i in range(num_agents):
            X = tf.compat.v1.greater_equal(weights[i][0], gate)  # 0.5 as gate
            X = tf.compat.v1.cast(X, dtype=tf.compat.v1.float32)
            matrix = tf.compat.v1.tile(X, multiples=[5])
            matrix = tf.compat.v1.cast(matrix, dtype=tf.compat.v1.float32)
            Y = tf.compat.v1.tile([1], multiples=[5])
            Y = tf.compat.v1.cast(Y, dtype=tf.compat.v1.float32)
            Y2 = tf.compat.v1.tile([0], multiples=[5])
            Y2 = tf.compat.v1.cast(Y2, dtype=tf.compat.v1.float32)

            # paket loss testing #
            if np.random.random() < packet_loss_rate:
                matrix2 = tf.compat.v1.tile([0], multiples=[5])
            else:
                matrix2 = tf.compat.v1.tile([1], multiples=[5])
            matrix2 = tf.compat.v1.cast(matrix2, dtype=tf.compat.v1.float32)
            matrix = tf.compat.v1.multiply(matrix, matrix2)

            Z = tf.compat.v1.concat([Y, matrix, matrix, matrix, matrix], 0)
            Z2 = tf.compat.v1.concat([Y, Y, Y, matrix, matrix], 0)
            Z3 = tf.compat.v1.concat([Y, Y, Y, Y, Y], 0)

            if i < 2:
                new_obs = tf.compat.v1.multiply(make_obs_ph_n[i], Z)
            elif i == 2:
                new_obs = tf.compat.v1.multiply(make_obs_ph_n[i], Z2)
            else:
                new_obs = tf.compat.v1.multiply(make_obs_ph_n[i], Z3)

            p_ = p_func(new_obs, int(act_pdtype_n[i].param_shape()[0]), scope="target_p_func_{}".format(i),
                       num_units=num_units)
            target_p_n.append(p_)

        target_p_func_vars = [U.scope_vars(U.absolute_scope_name("target_p_func_{}".format(i))) for i in range(num_agents)]

        target_var_list = []
        target_var_list.extend(target_p_func_vars)
        target_var_list = list(itertools.chain(*target_var_list))
        w_target_var_list = []
        w_target_var_list.extend(target_weights_vars)
        w_target_var_list = list(itertools.chain(*w_target_var_list))

        update_target_p = make_update_exp(var_list, target_var_list)
        update_target_p_w = make_update_exp(w_var_list, w_target_var_list)

        target_act_sample_n = [act_pdtype_n[i].pdfromflat(target_p_n[i]).sample() for i in range(num_agents)]
        target_act = U.function(inputs=obs_ph_n, outputs=[target_act_sample_n, target_weights])

        return act, train, train_w, update_target_p, update_target_p_w, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, weights, act_space_n, q_func, optimizer, gate, packet_loss_rate, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # create distribtuions

        num_agents = len(make_obs_ph_n)
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_{}".format(i)) for i in
                    range(len(act_space_n))]
        target_ph_n = [tf.compat.v1.placeholder(tf.compat.v1.float32, [None], name="target_{}".format(i)) for i in range(num_agents)]

        p_n = []
        for i in range(num_agents):
            X = tf.compat.v1.greater_equal(weights[i][0], gate)  # 0.5 as gate
            X = tf.compat.v1.cast(X, dtype=tf.compat.v1.float32)
            matrix = tf.compat.v1.tile(X, multiples=[5])
            matrix = tf.compat.v1.cast(matrix, dtype=tf.compat.v1.float32)
            Y = tf.compat.v1.tile([1], multiples=[5])
            Y = tf.compat.v1.cast(Y, dtype=tf.compat.v1.float32)
            Y2 = tf.compat.v1.tile([0], multiples=[5])
            Y2 = tf.compat.v1.cast(Y2, dtype=tf.compat.v1.float32)

            # paket loss testing #
            if np.random.random() < packet_loss_rate:
                matrix2 = tf.compat.v1.tile([0], multiples=[5])
            else:
                matrix2 = tf.compat.v1.tile([1], multiples=[5])
            matrix2 = tf.compat.v1.cast(matrix2, dtype=tf.compat.v1.float32)
            matrix = tf.compat.v1.multiply(matrix, matrix2)

            Z = tf.compat.v1.concat([Y, matrix, matrix, matrix, matrix], 0)
            Z2 = tf.compat.v1.concat([Y, Y, Y, matrix, matrix], 0)
            Z3 = tf.compat.v1.concat([Y, Y, Y, Y, Y], 0)

            if i < 2:
                p_n.append(tf.compat.v1.multiply(make_obs_ph_n[i], Z))
            elif i == 2:
                p_n.append(tf.compat.v1.multiply(make_obs_ph_n[i], Z2))
            else:
                p_n.append(tf.compat.v1.multiply(make_obs_ph_n[i], Z3))

        q_input = tf.compat.v1.concat(p_n + act_ph_n, 1)
        # if local_q_func:
        #     q_input = tf.compat.v1.concat([p_n[q_index], act_ph_n[q_index]], 1)
        # q_input = tf.compat.v1.concat(obs_ph_n + act_ph_n, 1)
        q_n = [q_func(q_input, 1, scope="q_func_{}".format(i), num_units=num_units)[:, 0] for i in range(num_agents)]
        q_func_vars = [U.scope_vars(U.absolute_scope_name("q_func_{}".format(i))) for i in range(num_agents)]
        q_loss_n = [tf.compat.v1.reduce_mean(tf.compat.v1.square(q - target_ph)) for q, target_ph in zip(q_n, target_ph_n)]
        q_loss = tf.compat.v1.reduce_sum(q_loss_n)
        loss = q_loss  # + 1e-3 * q_reg

        var_list = list(itertools.chain(*q_func_vars))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + weights + target_ph_n, outputs=loss, updates=[optimize_expr])
        print(obs_ph_n + act_ph_n + weights + target_ph_n)
        q_values = U.function(obs_ph_n + act_ph_n, q_n)

        # target network
        target_q_n = [q_func(q_input, 1, scope="target_q_func_{}".format(i), num_units=num_units)[:,0]
                      for i in range(num_agents)]

        target_q_func_vars = [U.scope_vars(U.absolute_scope_name("target_q_func_{}".format(i)))
                              for i in range(num_agents)]

        traget_var_list = list(itertools.chain(*target_q_func_vars))
        update_target_q = make_update_exp(var_list, traget_var_list)

        target_q_values = U.function(obs_ph_n + act_ph_n + weights, target_q_n)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


def q_w_train(make_obs_ph_n, weights, act_space_n, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        target_ph_n = [tf.compat.v1.placeholder(tf.compat.v1.float32, [None], name="w_target_{}".format(i)) for i in range(num_agents)]

        q_input = tf.compat.v1.concat(obs_ph_n + weights, 1)
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
        train = U.function(inputs=obs_ph_n + weights + target_ph_n, outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + weights, q_n)

        # target network
        target_q_n = [q_func(q_input, 1, scope="w_target_q_func_{}".format(i), num_units=num_units)[:, 0] for i in
                      range(num_agents)]
        target_q_func_vars = [U.scope_vars(U.absolute_scope_name("w_target_q_func_{}".format(i))) for i in
                              range(num_agents)]

        traget_var_list = list(itertools.chain(*target_q_func_vars))
        update_target_q = make_update_exp(var_list, traget_var_list)
        target_q_values = U.function(obs_ph_n + weights, target_q_n)
        return train, update_target_q, {'w_q_values': q_values, 'w_target_q_values': target_q_values}

class MADDPGAgentTrainer_C(AgentTrainer):
    def __init__(self, name, model, model_weights, obs_shape_n, act_space_n, args, gate, loss_rate, update_option,
                 local_q_func=False,
                 size_buffer=1e6):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args
        self.update_interval = args.update_interval
        obs_ph_n = []
        weights = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
            weights.append(U.BatchInput((1,), name="weights" + str(i)).get())
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            weights=weights,
            q_func=model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            gate=gate,
            packet_loss_rate=loss_rate
        )
        self.w_q_train, self.w_q_update, self.w_q_debug = q_w_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            weights=weights,
            q_func=model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.w_train, self.p_update, self.w_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            weights_ph=weights,
            p_W=model_weights,
            p_func=model,
            q_func=model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            gate=gate,
            packet_loss_rate=loss_rate
        )
        # Create experience buffer for timer
        self.replay_buffer = ReplayBuffer_W(size_buffer)
        #        self.replay_buffer = ReplayBuffer(args.batch_size)
        #        self.max_replay_buffer_len = args.batch_size
        #        50 * args.max_episode_len
        # args.batch_size = 64
        self.max_replay_buffer_len = args.batch_size * 2
        self.replay_sample_index = None
        self.update_option = update_option

    def action(self, obs_n):
        obs = [obs[None] for obs in obs_n]
        action_list = self.act(*(obs))
        return [act[0].tolist() for act in action_list[0]], [act[0][0] for act in action_list[1]]

    def experience(self, obs, act, weight, rew, com_rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, weight, rew, com_rew, new_obs, [float(d) for d in done])

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):

        # if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
        #     return
        if not t % 100 == 0:  # only update every 100 steps
            return
        if not self.update_option:
            return

        #        self.replay_sample_index = range(self.args.batch_size)
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        index = self.replay_sample_index
        samples = self.replay_buffer.sample_index(index)
        obs_n, act_n, weights_n, rew_n, com_rew_n, obs_next_n, done_n = [np.swapaxes(item, 0, 1) for item in samples]

        [target_act_next_n, target_w_next_n] = self.p_debug['target_act'](*(obs_next_n.tolist()))
        target_q_next_n = self.q_debug['target_q_values'](*(obs_next_n.tolist() + target_act_next_n + target_w_next_n))
        target_q_n = [rew + self.args.gamma * (1.0 - done) * target_q_next for rew, done, target_q_next in
                      zip(rew_n, done_n, target_q_next_n)]
        target_q_n = [target_q for target_q in target_q_n]

        # target_q_n = np.swapaxes(target_q_n, 0, 1)

        # print(obs_n.tolist() + act_n.tolist() + weights_n[0].tolist() + target_q_n)
        # print(np.array(obs_n.tolist()).shape)
        # print(np.array(act_n.tolist()).shape)
        # print(np.array(weights_n.tolist()).shape)
        # print(np.array(target_q_n).shape)

        q_loss = self.q_train(*(obs_n.tolist() + act_n.tolist() + weights_n.tolist() + target_q_n))

        target_q_next_n_w = self.w_q_debug['w_target_q_values'](*(obs_next_n.tolist() + target_w_next_n))
        target_q_n_w = [com_rew + self.args.gamma * (1.0 - done) * target_q_next for com_rew, done, target_q_next in
                        zip(com_rew_n, done_n, target_q_next_n_w)]
        target_q_n_w = [target_q for target_q in target_q_n_w]
        w_q_loss = self.w_q_train(*(obs_n.tolist() + weights_n.tolist() + target_q_n_w))

        p_loss = self.p_train(*(obs_n.tolist() + act_n.tolist()))
        w_p_loss = self.w_train(*(obs_n.tolist() + weights_n.tolist()))

        self.p_update()
        self.w_update()
        self.q_update()
        self.w_q_update()
        return [q_loss, p_loss, np.mean(rew_n)]

