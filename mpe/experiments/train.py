import sys
sys.path.append("..")
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
from experiments.Neural_Networks import get_trainers, get_action, stor_memeory
import trainer.common.tf_util as U

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # environment settings
    parser.add_argument("--scenario", choices=['simple_spread_partial', 'simple_tag_partial'],
                        default='simple_spread_partial', help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--obs-range", type=list, default=[0, 0, 2, 4, 5, 5, 5, 4, 4], help="observation range")
    parser.add_argument("--capacity", type=int, default=2, help="bandwidth limitation")
    parser.add_argument("--map-size", type=int, default=200, choices=[20, 60, 100, 150, 200],
                        help="map size in x*x m^2 which affect delay of comm.")
    parser.add_argument("--est-delay", type=float, default=250, choices=[20, 70, 120, 200, 250],
                        help="estimated max delay for normolization")

    # model parameters
    parser.add_argument("--trainer", choices=['DAMA', 'SchedNet', 'ATOC', 'MADDPG', 'GACML'],
                        default='DAMA', help="name of the trainer")
    parser.add_argument("--com_type", default='learn', choices=['no', 'full', 'learn'], help="type of com.")
    parser.add_argument("--model_name", default='maddpg_learn2com', help="description of the model")
    parser.add_argument("--delay_episodes", default=48000,
                        help="stop iteration after certain episodes in delay learning")
    parser.add_argument("--display", action="store_true", default=False, help="display on screen")
    parser.add_argument("--restore", action="store_true", default=False, help="load previous model")
    parser.add_argument("--update-interval", type=int, default=50, help="update every x steps")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--beta", type=float, default=0.05, help="discount factor")
    parser.add_argument("--dim-message", type=int, default=12, help="dimension of messages")
    # checkpointing
    parser.add_argument("--save_dir", type=str, default="./results/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=50,
                        help="save model once every time this many episodes are completed")
    return parser.parse_args()


def make_env(scenario_name, arglist, pack_size=1, map_size=100):
    from multiagent_praticle_envs.multiagent.environment import MultiAgentEnv
    import multiagent_praticle_envs.multiagent.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(arglist.obs_range)
    discrete_action = True  # if scenario_name == 'simple_spread' else False
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        scenario.get_centralnode_delay, discrete_action=discrete_action, pack_size=pack_size,
                        map_size=map_size, est_delay=arglist.est_delay)
    return env, world.num_adversaries

def train(arglist):
    with U.single_threaded_session():

        # Create environment
        pack_size = 1
        env, num_adversaries = make_env(arglist.scenario, arglist, pack_size, arglist.map_size)

        obs_shape_n = [(env.observation_space[i].shape[0]-num_adversaries,) for i in range(env.n)]
        obs_shape_net = [(num_adversaries,) for i in range(env.n)]

        arglist.n_good_agent = env.n - num_adversaries
        arglist.num_adversaries = num_adversaries

        trainers, trainers_adv, trainers_good = get_trainers(env, num_adversaries, obs_shape_n, obs_shape_net, arglist)
        print("----------------------trainer-----------------------")
        # Initialize
        U.initialize()
        savers = [tf.compat.v1.train.Saver(U.scope_vars(trainer.name)) for trainer in trainers]

        # Load previous results, if necessary
        if not os.path.exists(arglist.save_dir):
            os.mkdir(arglist.save_dir)
        model_path = arglist.save_dir + arglist.model_name + "/"
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        if arglist.restore:
            print('Loading previous state...')
            U.load_state(model_path)

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_com = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_com_weights = []
        final_ep_ag_rewards = []  # agent rewards for training curve
        catch_n_e = []
        final_catch_n_e = []
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.compat.v1.train.Saver()
        obs_n_all = env._reset()
        obsnet_n = [obs[-num_adversaries:] for obs in obs_n_all[:num_adversaries]]
        obs_n = [obs[:-num_adversaries] for obs in obs_n_all[:num_adversaries]]
        episode_step = 0
        train_step = 0
        catch_n = 0
        com_freq = [0]
        message_n = np.zeros([num_adversaries, arglist.dim_message])
        t_start = time.time()
        steps_num = []
        time_n_all, time_n_e = [], []
        central_delay = {0: 0, 20: 10+5, 60: 40, 100: 60, 150: 80, 200: 95}  # map_size: mean_delay, 5 for routing cost

        if arglist.restore:
            print('Loading previous state...')
            U.load_state(model_path)
            rew_file_name = model_path + arglist.model_name + '_rewards.pkl'
            f_rew = open(rew_file_name, 'rb')
            final_ep_rewards = pickle.load(f_rew)

            com_weights_file_name = model_path + arglist.model_name + '_com_weights.pkl'
            com_rew = open(com_weights_file_name, 'rb')
            final_ep_com_weights = pickle.load(com_rew)

            com_freq_file_name = model_path + arglist.model_name + '_com_freq.pkl'
            com_freq_rew = open(com_freq_file_name, 'rb')
            com_freq = pickle.load(com_freq_rew)

        print('Starting iterations...')
        while True:
            # get actions and necessary info
            action_n, time_n, message_n, weights_n = get_action(arglist, trainers_adv, obs_n, obsnet_n,
                                                                num_adversaries, message_n, env,
                                                                len(episode_rewards), central_delay[arglist.map_size])

            # set com gates and discounts
            gate = 0.5
            discount = 0.0
            if arglist.com_type == 'full':
                gate = -0.1
            elif arglist.com_type == 'no':
                gate = 1.1
            elif arglist.com_type == 'learn':
                gate = 0.5
                discount = 0.2
            mask_n = []
            com_rew_n = []

            time_n_step = np.copy(time_n).dot(arglist.est_delay/1000).tolist()
            time_n_e.append(np.mean(time_n_step)*1000)

            # step env
            new_obs_n_all, rew_n, done_n, info_n = env._step(action_n, time_n_step)

            # calculate com frequency
            com_count = 0
            if arglist.trainer in ['GACML', 'MADDPG']:
                for weight in weights_n:
                    if weight > gate:
                        mask_n.append(1)
                        com_count += 1
                    else:
                        mask_n.append(0)
                for i in range(len(rew_n)):
                    com_rew_n.append(rew_n[i] - discount * mask_n[i])
            ep_re = rew_n
            com_freq[-1] += com_count
            catch_n += env.world.catch

            new_obs_n = [obs[:-num_adversaries] for obs in new_obs_n_all[:num_adversaries]]
            new_obsnet_n = [obs[-num_adversaries:] for obs in new_obs_n_all[:num_adversaries]]
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # set mask
            if arglist.trainer in ['GACML', 'DAMA', 'ATOC', 'SchedNet']:
                view_0_mask = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                view_2_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]

                count = 0
                for i in range(len(new_obs_n)):
                    if count < 2:
                        new_obs_n[i] = np.multiply(view_0_mask, new_obs_n[i])
                    elif count == 2:
                        new_obs_n[i] = np.multiply(view_2_mask, new_obs_n[i])
                    count += 1

            # collect experience
            stor_memeory(arglist.trainer, trainers_adv, obs_n, action_n[:num_adversaries], rew_n, com_rew_n,
                             new_obs_n, done_n, terminal, obsnet_n, time_n[:num_adversaries], new_obsnet_n,
                             weights_n[:num_adversaries])

            obs_n = new_obs_n
            obsnet_n = new_obsnet_n

            for i, rew in enumerate(ep_re[:num_adversaries]):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                steps_num.append(episode_step)
                obs_n_all = env._reset()
                obsnet_n = [obs[-num_adversaries:] for obs in obs_n_all[:num_adversaries]]
                obs_n = [obs[:-num_adversaries] for obs in obs_n_all[:num_adversaries]]
                episode_step = 0
                episode_rewards.append(0)
                episode_com.append(0)
                for a in agent_rewards:
                    a.append(0)

                agent_info.append([[]])
                time_n_all.append(np.mean(time_n_e))
                time_n_e = []
                catch_n_e.append(catch_n/arglist.max_episode_len)
                catch_n = 0

            # increment global step counter
            train_step += 1

            if arglist.display:
                print("S:" + str(episode_step) + " Rewards: " + str(np.mean(rew_n[:num_adversaries])) +
                     " Timer:" + str(np.array(np.copy(time_n).dot(arglist.est_delay), dtype=int)))

                time.sleep(0.1)
                env._render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers_adv:
                agent.preupdate()
            
            for agent in trainers_adv:
                loss = agent.update(trainers_adv, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                final_ep_com_weights.append(np.mean(episode_com[-arglist.save_rate:]))
                final_catch_n_e.append(np.mean(catch_n_e[-arglist.save_rate:]))
                com_freq.append(0)
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}, catch_n: {}".format(
                    train_step, len(episode_rewards), round(final_ep_rewards[-1], 0),
                    round(time.time()-t_start, 0), round(np.mean(catch_n_e[-arglist.save_rate:]), 1)))

                t_start = time.time()
                if len(episode_rewards) % arglist.save_rate * 2000 == 0:

                    U.save_state(model_path, saver=saver)
                    [U.save_state(os.path.join(model_path, 'team_{}'.format(i)), saver=saver) for i, saver in
                        enumerate(savers)]
                rew_file_name = model_path + arglist.model_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                catch_n_file_name = model_path + arglist.model_name + '_catch_n.pkl'
                with open(catch_n_file_name, 'wb') as fp:
                    pickle.dump(final_catch_n_e, fp)
                com_weights_file_name = model_path + arglist.model_name + '_com_weights.pkl'
                with open(com_weights_file_name, 'wb') as fp:
                    pickle.dump(final_ep_com_weights, fp)
                com_freq_name = model_path + arglist.model_name + '_com_freq.pkl'
                with open(com_freq_name, 'wb') as fp:
                    pickle.dump(com_freq, fp)

                if str(arglist.model_name).split('_')[0] == 'delay learning':
                    if len(episode_rewards) > arglist.delay_episodes:
                        print("full com. ends at episodes " + str(arglist.delay_episodes))
                        break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)