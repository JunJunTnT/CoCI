import pickle
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import gym
import warnings
import argparse
import numpy as np
import tensorflow as tf
import maddpg.common.tf_util as U
import tf_slim.layers as layers
import highway_env
from experiments.Neural_Networks import get_trainers, get_action, stor_memeory
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_world_comm", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--update_interval", type=int, default=100)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)

    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def weight_model(input, num_outputs, scope, reuse=tf.compat.v1.AUTO_REUSE, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.selu)      # Xavier Initializer
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.selu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.sigmoid)
        return out

# == configuration ==
trainer_name = "DAMA"  # choose in [MADDPG, ACML, GACML, SchedNet, ATOC, DAMA]
scenario_name = "MADDPG full com"  # choose in [MADDPG no com, MADDPG full com, MADDPG learn2com, ACML, GACML, SchedNet, ATOC, DAMA]
loss_rate = 0.0     # set packet loss rate
# ==env settings
controlled_vehicles = 4
observation_dim = 5
mean_step = 50
view = 5
low_speed_range = [0, 3]
collision_punishment = -20
low_speed_punishment = 1
arrived_reward = 5
# ==
model_save_path = "./res/" + scenario_name + "/"
res_save_path = model_save_path
restore = False     # load previous model
display = False     # display on screen
update_option = True        # update model
# ===================

gate = 0        # set com gate based on scenario_name
if scenario_name in ['MADDPG full com', 'ACML', 'DAMA']:
    gate = -0.1
elif scenario_name in ['MADDPG no com']:
    gate = 1.1
else:
    gate = 0.5

def make_env():
    env = gym.make("intersection-multi-agent-v0")
    config = {
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "vehicles_count": view,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "type": "Kinematics",
                "absolute": True,
                "normalize": False,
            }
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "lateral": False,
                "longitudinal": True,
                "target_speeds": [0, 3, 6, 9]
            }
        },
        "controlled_vehicles": controlled_vehicles,
        # "initial_vehicle_count": 4,
        # "spawn_probability": 0,
        # "destination": "o1",
        "initial_vehicle_count": 6,
        "spawn_probability": 0,
        "screen_width": 1000,
        "screen_height": 1000,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5 * 1,
        "duration": 20  # [s]
    }
    env.config.update(config)
    env.reset()
    return env

def train(arglist):
    with U.single_threaded_session():

        env = make_env()

        obs_shape_n = []
        for _ in range(controlled_vehicles):
            obs_shape_n.append((view * observation_dim,))
        obsnet = [(4,) for _ in range(controlled_vehicles)]

        trainers = get_trainers(trainer_name, env, obs_shape_n, obsnet, arglist, gate, loss_rate, update_option)

        U.initialize()
        savers = [tf.compat.v1.train.Saver(U.scope_vars(trainer.name)) for trainer in trainers]

        episode_rewards = [0.0]  # sum of rewards for all agents
        mean_episode_reward = [0.0]
        mean_episode_step = [0.0]
        mean_episode_agent_reward = [[0.0 for _ in range(controlled_vehicles)]]
        loss_list = [0.0]
        action_buff = [0, 0, 0]
        speed_buff = [0.0 for _ in range(controlled_vehicles)]
        train_step = 0
        episode_step = 0
        step_in_this_win = 0
        arrive_count_in_this_win = [0 for _ in range(controlled_vehicles)]
        episode_agents_rewards = [0.0 for _ in range(controlled_vehicles)]
        crash_count = 0
        saver = tf.compat.v1.train.Saver()

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        if restore:
            U.load_state(model_save_path)
            rew_file_name = model_save_path + scenario_name + '_rewards.pkl'
            f_rew = open(rew_file_name, 'rb')
            mean_episode_reward = pickle.load(f_rew)


        while True:
            done_n = False
            obs_n = env.reset()

            obs_original = []
            count = 0
            for agent_obs in obs_n:
                agent_obs_vector = []
                for obs_item in agent_obs:
                    for index in obs_item:
                        agent_obs_vector.append(index)
                if not trainer_name == 'MADDPG':
                    if count < 2:
                        for i in range(5, 25):
                            agent_obs_vector[i] = 0
                    elif count == 2:
                        for i in range(15, 25):
                            agent_obs_vector[i] = 0
                obs_original.append(agent_obs_vector)
            obs_n = np.array(obs_original)

            has_arrived = [False for _ in range(controlled_vehicles)]
            obsnet = [np.array([0 for _ in range(controlled_vehicles)]) for _ in range(controlled_vehicles)]

            while not done_n:
                action_n = []
                action_n_with_percent, time_n, weights_n = get_action(trainer_name, trainers, obs_n, obsnet, len(episode_rewards))

                mask_n = []
                com_rew_n = []
                discount = 0

                action_net = [[0 for _ in range(3)] for _ in range(controlled_vehicles)]

                for item in action_n_with_percent:
                    action_n.append(np.argmax(item))

                for i in range(len(action_n)):
                    action_net[i][action_n[i]] = 1

                action_n = tuple(action_n)
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)


                new_obsnet = []
                for i in range(controlled_vehicles):
                    ego = new_obs_n[i]
                    tmp = []
                    for j in range(controlled_vehicles):
                        dist = (ego[j][1] - ego[0][1]) ** 2 + (ego[j][2] - ego[0][2]) ** 2
                        if dist < 0.4:
                            tmp.append(0.1)
                        else:
                            tmp.append(1)
                    new_obsnet.append(np.array(tmp))

                agents_speeds = (eval(str(info_n))).get("speed")
                agents_dones = (eval(str(info_n))).get("agents_dones")
                crasheds = (eval(str(info_n))).get("crashed")
                arrived = (eval(str(info_n))).get("arrived")

                crashed = False
                for elem in crasheds:
                    if elem:
                        crashed = True
                        break

                scaled_speed = []

                for speed in agents_speeds:
                    scaled_speed.append(U.lmap(speed, low_speed_range, [-1, 0]))
                step_agents_rewards = [0.0 for _ in range(controlled_vehicles)]

                if not crashed:
                    for i in range(controlled_vehicles):
                        episode_agents_rewards[i] += low_speed_punishment * np.clip(scaled_speed[i], -1, 0)
                        step_agents_rewards[i] += (low_speed_punishment * np.clip(scaled_speed[i], -1, 0))
                else:
                    crash_count += 1
                    for i in range(controlled_vehicles):
                        episode_agents_rewards[i] += crasheds[i] * collision_punishment
                        step_agents_rewards[i] += crasheds[i] * collision_punishment

                for i in range(controlled_vehicles):
                    if arrived[i] and not has_arrived[i]:
                        episode_agents_rewards[i] += arrived_reward
                        step_agents_rewards[i] += arrived_reward
                        arrive_count_in_this_win[i] += 1
                        has_arrived[i] = True

                obs_for_maddpg = []
                count = 0
                for agent_obs in new_obs_n:
                    agent_obs_vector = []

                    for obs_item in agent_obs:
                        for index in obs_item:
                            agent_obs_vector.append(index)

                    if not trainer_name == 'MADDPG':
                        if count < 2:
                            for i in range(5, 25):
                                agent_obs_vector[i] = 0
                        elif count == 2:
                            for i in range(15, 25):
                                agent_obs_vector[i] = 0
                    obs_for_maddpg.append(agent_obs_vector)
                obs_for_maddpg = np.array(obs_for_maddpg)

                time_ = [[t] for t in time_n]
                time_w = weights_n
                if trainer_name in ['MADDPG', 'GACML', 'ACML']:
                    for i in range(len(weights_n)):
                        if weights_n[i] > gate:
                            mask_n.append(1)
                        else:
                            mask_n.append(0)
                    for i in range(len(episode_agents_rewards)):
                        com_rew_n.append(step_agents_rewards[i] - discount * mask_n[i])

                stor_memeory(trainer_name, trainers, obs_n, action_n_with_percent, step_agents_rewards, com_rew_n
                             , obs_for_maddpg, agents_dones, obsnet, time_, time_w, new_obsnet, weights_n)

                obs_n = obs_for_maddpg
                obsnet = new_obsnet

                episode_step += 1
                train_step += 1
                step_in_this_win += 1

                for action in action_n:
                    action_buff[action] += 1
                for i in range(controlled_vehicles):
                    speed_buff[i] += agents_speeds[i]

                loss = None
                trainers[0].preupdate()
                loss = trainers[0].update(trainers[0], train_step)
                if loss is not None:
                    f_loss = open(model_save_path + "loss.txt", 'a+')
                    f_loss.write(str(loss) + "\n")

                if loss is not None:
                    loss_list.append(loss)

                if display:
                    env.render()
                for elem in step_agents_rewards:
                    episode_rewards[-1] += elem

                if all(arrived):
                    done_n = True

                if done_n:
                    mean_episode_reward[-1] += episode_rewards[-1] / controlled_vehicles
                    mean_episode_step[-1] += episode_step
                    mean_episode_agent_reward[-1] = np.sum([mean_episode_agent_reward[-1], episode_agents_rewards], axis=0)

                    episode_agents_rewards = [0.0 for _ in range(controlled_vehicles)]

                    if len(episode_rewards) % mean_step == 0:
                        mean_episode_reward[-1] /= mean_step
                        mean_episode_step[-1] /= mean_step

                        for elem in mean_episode_agent_reward[-1]:
                            elem /= mean_step
                        for elem in speed_buff:
                            elem /= step_in_this_win

                        print("mean_reward:" + str(mean_episode_reward[-1]) +
                              "\tstep:" + str(train_step) + "\tagent_reward:" + str(
                            mean_episode_agent_reward[-1]) + "\t crash_count:" + str(crash_count) +
                              "\taction:" + str(action_buff) + "\tspeed:" + str(speed_buff) +
                              "\tarrive_count:" + str(arrive_count_in_this_win))
                        total_arrive = 0
                        for elem in arrive_count_in_this_win:
                            total_arrive += elem

                        f_reward = open(res_save_path + scenario_name + "_reward.txt", 'a+')
                        f_reward.write(str(mean_episode_reward[-1]) + "\t" + str(crash_count / mean_step)
                                       + "\t" + str(total_arrive / mean_step / controlled_vehicles) + "\n")

                        crash_count = 0
                        step_in_this_win = 0
                        action_buff = [0, 0, 0]
                        speed_buff = [0.0 for _ in range(controlled_vehicles)]
                        mean_episode_step.append(0.0)
                        mean_episode_reward.append(0.0)
                        mean_episode_agent_reward.append([0.0 for _ in range(controlled_vehicles)])
                        arrive_count_in_this_win = [0 for _ in range(controlled_vehicles)]

                    if len(episode_rewards) % (mean_step * 10) == 0:
                        rew_file_name = model_save_path + scenario_name + '_rewards.pkl'
                        with open(rew_file_name, 'wb') as fp:
                            pickle.dump(mean_episode_reward, fp)

                        # buffer_file_name = model_save_path + scenario_name + '_buffer.pkl'
                        # with open(buffer_file_name, 'wb') as fp:
                        #     pickle.dump(trainers[0].replay_buffer, fp)

                        U.save_state(model_save_path, saver=saver)
                        [U.save_state(os.path.join(model_save_path, 'model'), saver=saver) for i, saver in
                         enumerate(savers)]
                    episode_rewards.append(0.0)
                    episode_step = 0

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
