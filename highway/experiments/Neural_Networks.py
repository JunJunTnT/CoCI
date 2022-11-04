import tf_slim.layers as layers
import tensorflow as tf

from maddpg.trainer.MADDPG import MADDPGAgentTrainer_C
from maddpg.trainer.SchedNet import SchednetTrainer
from maddpg.trainer.GACML import GACML_CTrainer
from maddpg.trainer.ACML import ACML_CTrainer
from maddpg.trainer.ATOC import ATOCTrainer
from maddpg.trainer.DAMA import TACAgentTrainer_PL

from maddpg.common.attention import multihead_attention
from gym import spaces


def get_action(trainer_name, trainers, obs_n, obsnet, Epi):
    action_n_with_percent = []
    time_n = []
    weights_n = []

    if trainer_name == 'DAMA':
        action_n_with_percent, time_n, weights_n = trainers[0].action(obs_n, obsnet, Epi)

    elif trainer_name == 'SchedNet':
        action_n_with_percent, weights_n, mask_n = trainers[0].action(obs_n)

    elif trainer_name in ['GACML', 'ACML']:
        action_n_with_percent, weights_n, mask_n = trainers[0].action(obs_n)

    elif trainer_name == 'ATOC':
        action_n_with_percent, weights_n, mask_n = trainers[0].action(obs_n)

    elif trainer_name =='MADDPG':
        action_n_with_percent, weights_n = trainers[0].action(obs_n)

    return action_n_with_percent, time_n, weights_n

def stor_memeory(trainer_name, trainers, obs_n, action_n_with_percent, step_agents_rewards, com_rew_n, new_obs_n, agents_dones,
                 obsnet, time_, time_w, new_obsnet,  weights_n):
        
    if trainer_name in ['GACML', 'ACML']:
        trainers[0].experience(obs_n, action_n_with_percent, [[t] for t in weights_n], step_agents_rewards, com_rew_n,
                               new_obs_n, agents_dones)

    elif trainer_name == 'DAMA':
        trainers[0].experience(obs_n, obsnet, time_, time_w, action_n_with_percent, step_agents_rewards, new_obs_n,
                                new_obsnet, agents_dones)

    elif trainer_name == 'SchedNet':
        trainers[0].experience(obs_n, action_n_with_percent, [[t] for t in weights_n], step_agents_rewards,
                               new_obs_n, agents_dones)

    elif trainer_name == 'MADDPG':
        trainers[0].experience(obs_n, action_n_with_percent, [[t] for t in weights_n], step_agents_rewards, com_rew_n,
                                new_obs_n, agents_dones)
    elif trainer_name == 'ATOC':
        trainers[0].experience(obs_n, action_n_with_percent, [[t] for t in weights_n], step_agents_rewards,
                               new_obs_n, agents_dones)


def get_trainers(trainer_name, env, obs_shape_n, obs_shape_net, arglist, gate, loss_rate, update_option):
    trainers = []
    if trainer_name == 'DAMA':
        trainers.append(TACAgentTrainer_PL(
            "DAMA", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, timer_model,
            obs_shape_n, obs_shape_net,
            env.action_space, [spaces.Discrete(4) for _ in range(4)], arglist))

    elif trainer_name == 'SchedNet':
        trainer = SchednetTrainer
        model_weights = weight_model
        trainers.append(trainer(
            "SchedNet", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, model_weights,
            obs_shape_n, env.action_space, arglist))

    elif trainer_name == 'GACML':
        trainer = GACML_CTrainer
        trainers.append(trainer(
            "GACML", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, weight_model,
            obs_shape_n, env.action_space, arglist, gate, loss_rate, update_option))

    elif trainer_name == 'MADDPG':
        trainer = MADDPGAgentTrainer_C
        trainers.append(trainer(
            "MADDPG", mlp_model, weight_model, obs_shape_n, env.action_space, arglist, gate, loss_rate, update_option,
            local_q_func=True))

    elif trainer_name == 'ATOC':
        trainer = ATOCTrainer
        trainers.append(trainer(
            "ATOC", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, weight_model,
            obs_shape_n, env.action_space, arglist))

    elif trainer_name == 'ACML':
        trainer = ACML_CTrainer
        trainers.append(trainer(
            "good_team", actor_mlp_I, attention_channel, actor_mlp_II, critic_mlp_model, weight_model,
            obs_shape_n, env.action_space, arglist))

    return trainers



def lstm_model(input_ph, num_outputs, scope, reuse=False, num_units=4):
    # print("Reusing LSTM_FC_MODEL: {}".format(reuse))
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        input_, c_, h_ = input_ph[:,:-2*num_units], input_ph[:,-2*num_units:-1*num_units], input_ph[:,-1*num_units:]
        out = tf.compat.v1.expand_dims(input_, 1)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.compat.v1.nn.relu)
        cell = tf.compat.v1.contrib.rnn.LSTMCell(num_units=num_units)
        state = tf.compat.v1.contrib.rnn.LSTMStateTuple(c_,h_)

        out, state = tf.compat.v1.nn.dynamic_rnn(cell, out, initial_state=state)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        state = tf.compat.v1.contrib.rnn.LSTMStateTuple(state.c, state.h)
        return out, state


def mlp_model(input, num_outputs, scope, reuse=tf.compat.v1.compat.v1.AUTO_REUSE, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
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
    

def tarmac_actor(input, num_outputs, scope, reuse=False, num_units=64, dim_message=4, rnn_cell=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.compat.v1.nn.relu)
        action = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        z = layers.fully_connected(out, num_outputs=dim_message, activation_fn=None)
        return action, z

def commnet_channel(inputs, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        h, c = inputs
        out_h = layers.fully_connected(h, num_outputs=num_units, activation_fn=None)
        out_c = layers.fully_connected(c, num_outputs=num_units, activation_fn=None)
        out = layers.fully_connected(out_h + out_c + h, num_outputs=num_units, activation_fn=None)
        return out

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


