from __future__ import print_function

import keras
from keras import initializers
import keras.backend as K
import tensorflow as tf
from keras import activations, constraints, initializers, regularizers
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, \
                        concatenate, Conv1D, BatchNormalization, CuDNNLSTM, Lambda, \
                        Multiply, Add, Activation, Flatten, LeakyReLU, InputSpec
from keras.engine import Layer
from keras.layers.wrappers import Wrapper, TimeDistributed

import numpy as np


class Alpha_Weights(Layer):
    def __init__(self, **kwargs):
        super(Alpha_Weights, self).__init__(**kwargs)

    def build(self, input_shape):
        # a trainable weight variable 
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2]), #input_shape[2]
                                      initializer='ones',  
                                      trainable=True)
        super(Alpha_Weights, self).build(input_shape)

    def call(self, x, **kwargs):
        # Shape x: (BATCH_SIZE, N, M)
        # Shape kernel: (N, M)
        # Shape output: (BATCH_SIZE, N, M)
        return self.kernel * x

    def compute_output_shape(self, input_shape):
        return input_shape


# a minus bias means the network is biased towards carry behavior in the initial stages
def Highway(value, n_layers, activation="tanh", gate_bias=-2):  
    dim = K.int_shape(value)[-1]
    bias = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):
        T_gate = Dense(units=dim, bias_initializer=bias, activation="sigmoid")(value)
        C_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(T_gate)
        transform = Dense(units=dim, activation=activation)(value)
        transform_gated = Multiply()([T_gate, transform])
        carry_gated = Multiply()([C_gate, value])
        value = Add()([transform_gated, carry_gated])
    return value

#-----------------------------------------------------------------------------------#

# from http://curlba.sh/jhartog/Mihail/blob/f19c455dcd804536a5895b5d5494119b4315e23b/lib/python2.7/site-packages/tensorflow/python/ops/gen_manip_ops.py

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
def roll(input, shift, axis, name=None):
  r"""Rolls the elements of a tensor along an axis.

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context
  if _ctx is None :#or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Roll", input=input, shift=shift, axis=axis, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Tshift", _op.get_attr("Tshift"),
              "Taxis", _op.get_attr("Taxis"))
    _execute.record_gradient(
      "Roll", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Roll", name,
        _ctx._post_execution_callbacks, input, shift, axis)
      return _result
    #except _core._FallbackException:
    #  return roll_eager_fallback(
    #      input, shift, axis, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

#-----------------------------------------------------------------------------------#

# https://github.com/JHart96/keras_gcn_sequence_labelling
class SpectralGraphConvolution(Layer):
    def __init__(self, output_dim,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None,
                 b_regularizer=None, bias=True, 
                 self_links=True, consecutive_links=True, 
                 backward_links=True, edge_weighting=False, **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node

        self.self_links = self_links
        self.consecutive_links = consecutive_links
        self.backward_links = backward_links
        self.edge_weighting = edge_weighting

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights

        self.input_dim = None
        self.W = None
        self.b = None
        self.num_nodes = None
        self.num_features = None
        self.num_relations = None
        self.num_adjacency_matrices = None

        super(SpectralGraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (None, features_shape[1], self.output_dim)
        return output_shape

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        self.input_dim = features_shape[1]
        self.num_nodes = features_shape[1]
        self.num_features = features_shape[2]
        self.num_relations = len(input_shapes) - 1

        self.num_adjacency_matrices = self.num_relations

        if self.consecutive_links:
            self.num_adjacency_matrices += 1

        if self.backward_links:
            self.num_adjacency_matrices *= 2

        if self.self_links:
            self.num_adjacency_matrices += 1

        self.W = []
        self.W_edges = []
        for i in range(self.num_adjacency_matrices):
            self.W.append(self.add_weight((self.num_features, self.output_dim), # shape: (num_features, output_dim)
                                                    initializer=self.init,
                                                    name='{}_W_rel_{}'.format(self.name, i),
                                                    regularizer=self.W_regularizer))

            if self.edge_weighting:
                self.W_edges.append(self.add_weight((self.input_dim, self.num_features), # shape: (num_features, output_dim)
                                                        initializer='ones',
                                                        name='{}_W_edge_{}'.format(self.name, i),
                                                        regularizer=self.W_regularizer))

        self.b = self.add_weight((self.input_dim, self.output_dim),
                                        initializer='random_uniform',
                                        name='{}_b'.format(self.name),
                                        regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(SpectralGraphConvolution, self).build(input_shapes)

    def call (self, inputs, mask=None):
        features = inputs[0] # Shape: (None, num_nodes, num_features)
        A = inputs[1:]  # Shapes: (None, num_nodes, num_nodes)

        eye = A[0] * K.zeros(self.num_nodes, dtype='float32') + K.eye(self.num_nodes, dtype='float32')

        # eye = K.eye(self.num_nodes, dtype='float32')

        if self.consecutive_links:
            #shifted = tf.manip.roll(eye, shift=1, axis=0)
            #shifted = tf.roll(eye, shift=1, axis=0)
            #shifted = roll(eye, shift=1, axis=0)
            #####################################################
            eye_len = eye.get_shape().as_list()[0] 
            #shifted = tf.concat((eye, eye), axis=0)
            #shifted = tf.concat((eye[eye_len-1: , :], eye[:eye_len-1 , :]), axis=0)
            shifted = tf.concat((eye[-1: , :], eye[:-1 , :]), axis=0)

            #####################################################

            A.append(shifted)

        if self.backward_links:
            for i in range(len(A)):
                A.append(K.permute_dimensions(A[i], [0, 2, 1]))

        if self.self_links:
            A.append(eye)

        AHWs = list()
        for i in range(self.num_adjacency_matrices):
            if self.edge_weighting:
                features *= self.W_edges[i]
            HW = K.dot(features, self.W[i]) # Shape: (None, num_nodes, output_dim)
            AHW = K.batch_dot(A[i], HW) # Shape: (None, num_nodes, num_features) --> (None, num_nodes, output_dim)
            AHWs.append(AHW)
        AHWs_stacked = K.stack(AHWs, axis=1) # Shape: (None, num_supports, num_nodes, num_features) --> Shape:(None, num_supports, num_nodes, output_dim)
        output = K.max(AHWs_stacked, axis=1) # Shape: (None, num_nodes, output_dim)

        if self.bias:
            output += self.b
        return self.activation(output)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'num_bases': self.num_bases,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



#-----------------------------------------------------------------------------------#

# from https://github.com/CyberZHG/keras-gcn
class GraphLayer(keras.layers.Layer):

    def __init__(self,
                 step_num=1,
                 activation=None,
                 **kwargs):
        """Initialize the layer.

        :param step_num: Two nodes are considered as connected if they could be reached in `step_num` steps.
        :param activation: The activation function after convolution.
        :param kwargs: Other arguments for parent class.
        """
        self.supports_masking = True
        self.step_num = step_num
        self.activation = keras.activations.get(activation)
        self.supports_masking = True
        super(GraphLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'step_num': self.step_num,
            'activation': self.activation,
        }
        base_config = super(GraphLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _get_walked_edges(self, edges, step_num):
        """Get the connection graph within `step_num` steps

        :param edges: The graph in single step.
        :param step_num: Number of steps.
        :return: The new graph that has the same shape with `edges`.
        """
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(K.batch_dot(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return K.cast(K.greater(deeper, 0.0), K.floatx())

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = self.activation(self._call(features, edges))
        return outputs

    def _call(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')


class GraphConv(GraphLayer):
    """Graph convolutional layer.

    h_i^{(t)} = \sigma \left ( \frac{ G_i^T (h_i^{(t - 1)} W + b)}{\sum G_i}  \right )
    """

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param units: Number of new states. If the input shape is (batch_size, node_num, feature_len), then the output
                      shape is (batch_size, node_num, units).
        :param kernel_initializer: The initializer of the kernel weight matrix.
        :param kernel_regularizer: The regularizer of the kernel weight matrix.
        :param kernel_constraint:  The constraint of the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: The initializer of the bias vector.
        :param bias_regularizer: The regularizer of the bias vector.
        :param bias_constraint: The constraint of the bias vector.
        :param kwargs: Other arguments for parent class.
        """
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.W, self.b = None, None
        super(GraphConv, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'use_bias': self.use_bias,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'bias_constraint': self.bias_constraint,
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = input_shape[0][2]
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def _call(self, features, edges):
        features = K.dot(features, self.W)
        if self.use_bias:
            features += self.b
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features) \
            / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())


class GraphPool(GraphLayer):

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask[0]


class GraphMaxPool(GraphPool):

    NEG_INF = -1e38

    def _call(self, features, edges):
        node_num = K.shape(features)[1]
        features = K.tile(K.expand_dims(features, axis=1), K.stack([1, node_num, 1, 1])) \
            + K.expand_dims((1.0 - edges) * self.NEG_INF, axis=-1)
        return K.max(features, axis=2)


class GraphAveragePool(GraphPool):

    def _call(self, features, edges):
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features) \
            / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())

#---------------------------------------------------------------------------------------------#

# from: https://github.com/danielegrattarola/keras-gat
class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape