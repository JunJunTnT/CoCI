import tf_slim.layers as layers
import tensorflow as tf

from trainer.MADDPG import MADDPGAgentTrainer_C
from trainer.SchedNet import SchednetTrainer
from trainer.DAMA import TACAgentTrainer_PL
from trainer.GACML import GACML_CTrainer
from trainer.ATOC import ATOCTrainer
from trainer.common.attention import multihead_attention
import numpy as np


def get_action(arglist, trainers, obs_n, obsnet_n, num_adversaries, message_n, env, Epi, C_delay=None):
    weights = []
    trainer_name = arglist.trainer

    if trainer_name == 'DAMA':
        action_n, time_n, weights = trainers[0].action(obs_n, obsnet_n, Epi)

    elif trainer_name == 'SchedNet':
        action_n, weights, mask = trainers[0].action(obs_n)
        if C_delay is not None:
            time_n = np.max(np.multiply(obsnet_n,mask),1)*arglist.capacity/num_adversaries + C_delay/arglist.est_delay# 4 ms rournd routing from central to users
        else:
            dalay_n = env._get_central_delay(num_adversaries, pack_size=1, map_size=arglist.map_size)   # 0.8 discount for  messages 
            D_c = np.max(np.multiply(dalay_n, mask))
            time_n = np.max(np.multiply(obsnet_n,mask),1)*arglist.capacity/num_adversaries + D_c
        time_n = time_n.tolist()

    elif trainer_name == 'GACML':
        action_n, weights, mask = trainers[0].action(obs_n)
        time_n = [0] * num_adversaries

    elif trainer_name == 'ATOC':
        action_n, weights, mask = trainers[0].action(obs_n)
        if C_delay is not None: 
            timer = C_delay/arglist.est_delay
            time_n = [timer*mask[i] for i in range(num_adversaries)]
        else:
            dalay_n = env._get_central_delay(num_adversaries, pack_size=1, map_size=arglist.map_size)  # the factor n and p are not used
            D_c = np.max(dalay_n)
            timer = D_c+dalay_n
            time_n = timer*mask

    elif trainer_name =='MADDPG':
        time_n = [0]*num_adversaries
        action_n, weights = trainers[0].action(obs_n)

    return action_n, time_n, message_n, weights

def stor_memeory(trainer_name, trainers, obs_n, action_n, rew_n, com_rew_n, new_obs_n, done_n, terminal,
                 obsnet_n, time_n, new_obsnet_n, weights_n):

    if trainer_name == 'DAMA':
        time_ = [ [t] for t in time_n]
        time_w = weights_n
        memory = [obs_n, obsnet_n, time_, time_w, action_n, rew_n, new_obs_n, new_obsnet_n, done_n]
        trainers[0].experience(*(memory))

    elif trainer_name == 'SchedNet':
        memory = [obs_n, action_n, [[t] for t in weights_n], rew_n, new_obs_n, done_n]
        trainers[0].experience(*(memory))

    elif trainer_name in ['MADDPG', "GACML"]:
        memory = [obs_n, action_n, [[t] for t in weights_n], rew_n, com_rew_n, new_obs_n, done_n]
        trainers[0].experience(*(memory))

    # else:
    #     for i, agent in enumerate(trainers):
    #         agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])



def get_trainers(env, num_adversaries, obs_shape_n, obs_shape_net, arglist):
    trainers = []
    ag_good, ag_adv = [], []

    if arglist.trainer == 'DAMA':
        trainer = TACAgentTrainer_PL
        Timer_Net = timer_model
        if num_adversaries > 0:
            trainers.append(trainer(
                "adversary_team", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, Timer_Net,
                obs_shape_n[:num_adversaries], obs_shape_net[-num_adversaries:],
                env.action_space[:num_adversaries], env.timer_w_space[:num_adversaries], arglist))
            ag_adv = [trainers[0]]
        if env.n - num_adversaries > 0:
            trainers.append(trainer(
                "good_team", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, Timer_Net,
                obs_shape_n[num_adversaries:], [(env.n - num_adversaries ,) for i in range(env.n - num_adversaries )], 
                env.action_space[num_adversaries:], env.timer_w_space[num_adversaries:],arglist))
            [obs[-num_adversaries:] for obs in obs_shape_net[-num_adversaries:]]
            ag_good = [trainers[1]]
            
    elif arglist.trainer == 'SchedNet':
        trainer = SchednetTrainer
        model_weights = weight_model
        if num_adversaries > 0:
            trainers.append(trainer(
                "schednet_team", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, model_weights,
                obs_shape_n[:num_adversaries], env.action_space[:num_adversaries], arglist))
            ag_adv = [trainers[0]]
        if env.n - num_adversaries > 0:
            trainers.append(trainer(
                "adv_schednet_team", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, model_weights,
                obs_shape_n[-num_adversaries:], env.action_space[-num_adversaries:], arglist))
            ag_good = [trainers[1]]

    elif arglist.trainer == 'GACML':
        trainer = GACML_CTrainer
        model_weights = weight_model
        if num_adversaries > 0:
            trainers.append(trainer(
                "schednet_team", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, model_weights,
                obs_shape_n[:num_adversaries], env.action_space[:num_adversaries], arglist))
            ag_adv = [trainers[0]]
        if env.n - num_adversaries > 0:
            trainers.append(trainer(
                "adv_schednet_team", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, model_weights,
                obs_shape_n[-num_adversaries:], env.action_space[-num_adversaries:], arglist))
            ag_good = [trainers[1]]

    elif arglist.trainer == 'MADDPG':
        trainer = MADDPGAgentTrainer_C
        model = mlp_model
        model_weights = weight_model
        if num_adversaries > 0:
            trainers.append(trainer(
                "maddpg_C", model, model_weights, obs_shape_n[:num_adversaries], env.action_space[:num_adversaries],
                arglist, arglist.lr,
                local_q_func=(arglist.trainer == 'ddpg')))
        if env.n - num_adversaries >0:
            trainers.append(trainer(
                "maddpg_C", model, model_weights, obs_shape_n[:num_adversaries], env.action_space[:num_adversaries],
                arglist, arglist.lr,
                local_q_func=(arglist.trainer == 'ddpg')))
        ag_adv, ag_good = trainers[:num_adversaries], trainers[num_adversaries:]

    elif arglist.trainer == 'ATOC':
        trainer = ATOCTrainer
        model_weights = weight_model
        if num_adversaries > 0:
            trainers.append(trainer(
                "ATOC", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, model_weights,
                obs_shape_n[:num_adversaries], env.action_space[:num_adversaries], arglist))
            ag_adv = [trainers[0]]
        if env.n - num_adversaries > 0:
            trainers.append(trainer(
                "ATOC_good", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, model_weights,
                obs_shape_n[-num_adversaries:], env.action_space[-num_adversaries:], arglist))
            ag_good = [trainers[1]]
    else:
        raise NotImplementedError
    return trainers, ag_adv, ag_good

def mlp_model(input, num_outputs, scope, reuse=tf.compat.v1.compat.v1.AUTO_REUSE, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    # print("Reusing mlp_MODEL: {}".format(reuse))
    with tf.compat.v1.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def timer_model(input, num_outputs, scope, reuse=tf.compat.v1.AUTO_REUSE, num_units=32, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.selu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.selu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
#                                     tf.compat.v1.nn.sigmoid)
        return out
    
def weight_model(input, num_outputs, scope, reuse=tf.compat.v1.AUTO_REUSE, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.selu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.selu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.compat.v1.nn.sigmoid)
        return out

def critic_mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out
    
def attention_channel(input, scope, reuse=False, num_heads=8):
    return multihead_attention(input, input, input,
                               num_heads=num_heads,
                               dropout_rate=0,
                               training=True,
                               causality=False,
                               scope=scope)

    
def attention_channel_index(input, query, scope, reuse=False, num_heads=8):
    return multihead_attention(query, input, input,
                               num_heads=num_heads,
                               dropout_rate=0,
                               training=True,
                               causality=False,
                               scope=scope)

def actor_mlp_I(input, num_outputs, scope, reuse=tf.compat.v1.AUTO_REUSE, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.compat.v1.nn.relu)
        return out
    
def actor_mlp_II(input, num_outputs, scope, reuse=tf.compat.v1.AUTO_REUSE, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def channel(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        z_mu = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
        z_log_sigma_sq = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        eps = tf.compat.v1.random_normal(
            shape=tf.compat.v1.shape(z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.compat.v1.float32)
        z = z_mu + tf.compat.v1.exp(0.5 * z_log_sigma_sq) * eps
        return z, z_mu, z_log_sigma_sq
