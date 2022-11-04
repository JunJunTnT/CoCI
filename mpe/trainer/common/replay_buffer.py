import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_C(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, weight, reward, com_rew, obs_tp1, done):
        data = (obs_t, action, weight, reward, com_rew, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, weights, rewards, com_rewards, obses_tp1, dones = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, weight, reward, com_reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            weights.append(np.array(weight, copy=False))
            rewards.append(reward)
            com_rewards.append(com_reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(weights), np.array(rewards), \
            np.array(com_rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)



class ReplayBuffer_W(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, weight, reward,  obs_tp1, done):
        data = (obs_t, action, weight, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    

    def _encode_sample(self, idxes):
        obses_t, actions, weights, rewards, obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, weight, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            weights.append(np.array(weight, copy=False))
            rewards.append(reward)
            # com_rewards.append(com_reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(weights), np.array(rewards), \
             np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_Packet_loss(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, pre_obs, obs_t, action, weight, reward, com_rew, obs_tp1, done):
        data = (pre_obs, obs_t, action, weight, reward, com_rew, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        pre_obses, obses_t, actions, weights, rewards, com_rewards, obses_tp1, dones = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            pre_obs, obs_t, action, weight, reward, com_reward, obs_tp1, done = data
            pre_obses.append(np.array(pre_obs, copy=False))
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            weights.append(np.array(weight, copy=False))
            rewards.append(reward)
            com_rewards.append(com_reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(pre_obses), np.array(obses_t), np.array(actions), np.array(weights), np.array(rewards), \
               np.array(com_rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_N(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, obsnet_t, action, reward, obs_tp1, obsnet_tp1, done):
        data = (obs_t, obsnet_t, action, reward, obs_tp1, obsnet_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    

    def _encode_sample(self, idxes):
        obses_t, obsnetes_t, actions, rewards, obses_tp1, obsnetes_tp1, dones = [], [], [], [], [], [],[]
        for i in idxes:
            data = self._storage[i]
            obs_t, obsnet_t, action, reward, obs_tp1, obsnet_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            obsnetes_t.append(np.array(obsnet_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            obsnetes_tp1.append(np.array(obsnet_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(obsnetes_t), np.array(actions), np.array(rewards), \
                np.array(obses_tp1), np.array(obsnetes_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)



class ReplayBuffer_R(object):
    def __init__(self, size, experience_size=15):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.experience_size = experience_size
        self.first_ever_episode = True
        self.q_lstm_on = True
        self.p_lstm_on = True


    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    # def add(self, obs, act, weight, rew,
    #                                  new_obs, done,  # terminal,
    #                                  p_in_c_n, p_in_h_n,
    #                                  p_out_c_n, p_out_h_n,
    #                                  q_in_c_n, q_in_h_n,
    #                                  q_out_c_n, q_out_h_n, new_episode):
    #     data = [obs, act, weight, rew, new_obs, done,  # terminal,
    #                                  p_in_c_n, p_in_h_n,
    #                                  p_out_c_n, p_out_h_n,
    #                                  q_in_c_n, q_in_h_n,
    #                                  q_out_c_n, q_out_h_n, new_episode]
        # assert len(args) == self.experience_size, "[Replay Buffer] Got exp size {}, but expected size {}".format(len(args),self.experience_size)
        # buffer: [[(episode exp, ...), (episode exp),... ], [(diff ep...), (...)]]
        # data = (obs_t, action, weight, reward, obs_tp1, done,
        #             p_c_in, p_h_in, p_c_out, p_h_out,
        #             q_c_in, q_h_in, q_c_out, q_h_out)
        # data = args[:-1]
    def add(self, obs, act, weight, rew,
            new_obs, done,  # terminal,
            p_in_c_n, p_in_h_n,
            p_out_c_n, p_out_h_n,
            new_episode):

        data = [obs, act, weight, rew, new_obs, done,  # terminal,
                                     p_in_c_n, p_in_h_n,
                                     p_out_c_n, p_out_h_n,
                                     new_episode]
        # print(type(data))
        new_episode = data[-1]
        data = data [:-1]
        # print(len(data))
        # new_episode = args[-1]
        if new_episode:
            if self.first_ever_episode:
                self.first_ever_episode = False
            else:
                self._next_idx = (self._next_idx + 1) % self._maxsize


        if self._next_idx >= len(self._storage):
            # still have room for more episodes
            if new_episode:
                # start a new episode list @ storage[next_ind]
                self._storage.append([data])
            else:
                # add onto current episode length
                self._storage[self._next_idx].append(data)
        else: # replace old data
            if new_episode:
                self._storage[self._next_idx] = [data]
            else:
                self._storage[self._next_idx].append(data)

    def _encode_sample(self, idxes):
        obses_t, actions, weights, rewards, obses_tp1, dones = [], [], [], [], [], []
        p_c_in, p_h_in= [], []
        p_c_out, p_h_out = [], []
        q_c_in, q_h_in= [], []
        q_c_out, q_h_out = [], []
        for i in idxes:
            data = self._storage[i]
            # print(len(data))
            for el in data:
                # print(len(el))
                # print(el)

                # obs_t, action, weight, reward, obs_tp1, done, p_c_in_t, p_h_in_t, p_c_out_t, p_h_out_t, q_c_in_t, q_h_in_t, q_c_out_t, q_h_out_t = el
                obs_t, action, weight, reward, obs_tp1, done, p_c_in_t, p_h_in_t, p_c_out_t, p_h_out_t = el

                # might need to change the dimensions on obs and actions
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                weights.append(np.array(weight, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)

                if self.p_lstm_on:
                    p_c_in.append(p_c_in_t)
                    p_h_in.append(p_h_in_t)
                    p_c_out.append(p_c_out_t)
                    p_h_out.append(p_h_out_t)
                # if self.q_lstm_on:
                #     q_c_in.append(q_c_in_t)
                #     q_h_in.append(q_h_in_t)
                #     q_c_out.append(q_c_out_t)
                #     q_h_out.append(q_h_out_t)
            # Check the dimensions
        return np.array(obses_t), np.array(actions), np.array(weights), np.array(rewards), np.array(obses_tp1), np.array(dones), \
               np.array(p_c_in),np.array(p_h_in),np.array(p_c_out),np.array(p_h_out)

        # return np.array(obses_t), np.array(actions), np.array(weights), np.array(rewards), np.array(obses_tp1), np.array(dones), \
        #        np.array(p_c_in), np.array(p_h_in), np.array(p_c_out), np.array(p_h_out), np.array(q_c_in), np.array(q_h_in), np.array(q_c_out), np.array(q_h_out)

    def make_index_lstm(self, batch_size):
        res = []
        for _ in range(batch_size):
            rand_int = random.randint(0, len(self._storage) -1)
            res.append(rand_int)
        return res

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_W_copy(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, weight, reward, obs_tp1, done):
        data = (obs_t, action, weight, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, weights, rewards,  obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, weight, reward,  obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            weights.append(np.array(weight, copy=False))
            rewards.append(reward)

            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(weights), np.array(rewards), \
               np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_D(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs, obsnet, time, time_w, act, rew, new_obs, new_obsnet, done):
        data = (obs, obsnet, time, time_w, act, rew, new_obs, new_obsnet, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, obsnetes_t, actions, rewards, obses_tp1, obsnetes_tp1, \
        dones = [], [], [], [], [], [], []
        times, times_w = [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, obsnet_t, time, time_w, action, reward, obs_tp1, obsnet_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            obsnetes_t.append(np.array(obsnet_t, copy=False))
            #            timer_n_matrix.append(np.array(time_matrix, copy=False))
            #            messages.append(np.array(message, copy=False))
            #            message_ns.append(np.array(message_n, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            obsnetes_tp1.append(np.array(obsnet_tp1, copy=False))
            #            timer_n_matrix_tp1.append(np.array(new_time_matrix, copy=False))
            dones.append(done)
            #            messages_nexts.append(message_next)
            times.append(time)
            #            time_nexts.append(time_next)
            times_w.append(time_w)
        #            time_nexts_w.append(time_w_next)
        return np.array(obses_t), np.array(obsnetes_t), np.array(times), np.array(times_w), np.array(actions), \
               np.array(rewards), np.array(obses_tp1), np.array(obsnetes_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
