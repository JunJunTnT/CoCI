
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''

import numpy as np
import tensorflow as tf


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.compat.v1.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.compat.v1.get_variable("beta", params_shape, initializer=tf.compat.v1.zeros_initializer())
        gamma = tf.compat.v1.get_variable("gamma", params_shape, initializer=tf.compat.v1.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.compat.v1.variable_scope("shared_weight_matrix"):
        embeddings = tf.compat.v1.get_variable('weight_mat',
                                     dtype=tf.compat.v1.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.compat.v1.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.compat.v1.concat((tf.compat.v1.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.compat.v1.matmul(Q, tf.compat.v1.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # softmax
        outputs = tf.compat.v1.nn.softmax(outputs)
        attention = tf.compat.v1.transpose(outputs, [0, 2, 1])
        tf.compat.v1.summary.image("attention", tf.compat.v1.expand_dims(attention[:1], -1))

        # dropout
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.compat.v1.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Linear projections
        Q = tf.compat.v1.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.compat.v1.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.compat.v1.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.compat.v1.concat(tf.compat.v1.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.compat.v1.concat(tf.compat.v1.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.compat.v1.concat(tf.compat.v1.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.compat.v1.concat(tf.compat.v1.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Normalize
        outputs = ln(outputs)

    return outputs


def multihead_attention_with_weight(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Linear projections
        Q = tf.compat.v1.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.compat.v1.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.compat.v1.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.compat.v1.concat(tf.compat.v1.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.compat.v1.concat(tf.compat.v1.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.compat.v1.concat(tf.compat.v1.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.compat.v1.concat(tf.compat.v1.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Normalize
        outputs = ln(outputs)

    return outputs, V

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Inner layer
        outputs = tf.compat.v1.layers.dense(inputs, num_units[0], activation=tf.compat.v1.nn.relu)

        # Outer layer
        outputs = tf.compat.v1.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf.compat.v1
    inputs = tf.compat.v1.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.compat.v1.float32)

    outputs = label_smoothing(inputs)

    with tf.compat.v1.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.compat.v1.shape(inputs)[0], tf.compat.v1.shape(inputs)[1]  # dynamic
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # position indices
        position_ind = tf.compat.v1.tile(tf.compat.v1.expand_dims(tf.compat.v1.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.compat.v1.convert_to_tensor(position_enc, tf.compat.v1.float32)  # (maxlen, E)

        # lookup
        outputs = tf.compat.v1.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.compat.v1.where(tf.compat.v1.equal(inputs, 0), inputs, outputs)

        return tf.compat.v1.to_float(outputs)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.compat.v1.cast(global_step + 1, dtype=tf.compat.v1.float32)
    return init_lr * warmup_steps ** 0.5 * tf.compat.v1.minimum(step * warmup_steps ** -1.5, step ** -0.5)


if __name__ == "__main__":

  queries = tf.compat.v1.random.normal([3, 1, 8]) #[batch, num_agent, mess_dim]
  key = tf.compat.v1.random.normal([3, 10, 8]) ##[batch, num_agent, mess_dim]
  values = key
  output = multihead_attention(queries, key, values, num_heads=1)
#  output = multihead_attention_with_weight(queries, key, values, num_heads=1)
  
  
#  def multihead_attention(queries, keys, values,
#                        num_heads=8,
#                        dropout_rate=0,
#                        training=True,
#                        causality=False,
#                        scope="multihead_attention"):
#    '''Applies multihead attention. See 3.2.2
#    queries: A 3d tensor with shape of [N, T_q, d_model].
#    keys: A 3d tensor with shape of [N, T_k, d_model].
#    values: A 3d tensor with shape of [N, T_k, d_model].
#    num_heads: An int. Number of heads.
#    dropout_rate: A floating point number.
#    training: Boolean. Controller of mechanism for dropout.
#    causality: Boolean. If true, units that reference the future are masked.
#    scope: Optional scope for `variable_scope`.#
#    Returns
#      A 3d tensor with shape of (N, T_q, C)  

              
  print(f'Input shape: {values.shape}')
  print(output)
  print(tf.compat.v1.trainable_variables())
