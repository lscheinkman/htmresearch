# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import tensor_shape, common_shapes, ops
from tensorflow.python.keras import regularizers, activations, initializers, constraints
from tensorflow.python.ops import gen_sparse_ops



class SparseWeightNN(keras.layers.Layer):
  """Similar to keras.layers.Dense but using sparse tensors.

  `SparseNN` implements the same operation as `keras.layers.Dense` using sparse
  weights and bias variables.

  :param units: dimensionality of the dense output space.
  :param activation: Activation function to use. If you don't specify anything,
                     no activation is applied
  :param kernel_initializer: Initializer for the `kernel` weights matrix.
  :param bias_initializer: Initializer for the bias vector.
  :param kernel_regularizer: Regularizer function applied to the `kernel`
                             weights matrix.
  :param bias_regularizer: Regularizer function applied to the bias vector.
  :param activity_regularizer: Regularizer function applied to the output of the
                               layer (its "activation")..
  :param sparsity: Percentage of weights that are allowed to be non-zero.
  """


  def __init__(self,
               units,
               sparsity=0.5,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

    assert (sparsity < 1.0)
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(SparseWeightNN, self).__init__(
      activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.sparsity = sparsity
    self.nonZeros = int(round(self.units * self.sparsity))


  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `SparseWeightNN` '
                       'should be defined. Found `None`.')
    input_size = input_shape[-1].value
    self.dense_shape = [input_size, self.units]

    # Compute sparse indices
    idx = np.stack(np.indices(self.dense_shape), axis=-1)
    map(np.random.shuffle, idx)
    idx = idx[:, :self.nonZeros]
    idx = np.sort(idx, axis=1)
    self.indices = np.reshape(idx, (-1, 2))

    # Only compute weight for the nonzero values
    self.kernel = self.add_weight(
      'kernel',
      shape=[input_size * self.nonZeros, ],
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      dtype=self.dtype,
      trainable=True)

    self.bias = self.add_weight(
      'bias',
      shape=[self.units, ],
      initializer=self.bias_initializer,
      regularizer=self.bias_regularizer,
      constraint=self.bias_constraint,
      dtype=self.dtype,
      trainable=True)

    with tf.name_scope('summaries'):
      tf.summary.histogram('kernel', self.kernel)
      tf.summary.histogram('bias', self.bias)

    self.built = True


  def call(self, inputs, **kwargs):
    with tf.name_scope("SparseWeightNN"):
      inputs = ops.convert_to_tensor_or_indexed_slices(inputs, dtype=self.dtype)
      rank = common_shapes.rank(inputs)
      if rank > 2:
        raise NotImplementedError('Input vector must not have more than 2 dimensions')

      outputs = gen_sparse_ops.sparse_tensor_dense_mat_mul(a_indices=self.indices,
                                                           a_values=self.kernel,
                                                           a_shape=self.dense_shape,
                                                           b=inputs,
                                                           adjoint_a=True,
                                                           adjoint_b=True)
      outputs = tf.transpose(outputs)
      if self.use_bias:
        outputs = tf.nn.bias_add(outputs, self.bias)
      if self.activation is not None:
        return self.activation(outputs)  # pylint: disable=not-callable
      return outputs


  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
        'The innermost dimension of input_shape must be defined, but saw: %s'
        % input_shape)
    return input_shape[:-1].concatenate(self.units)


  def get_config(self):
    config = {
      'units': self.units,
      'activation': activations.serialize(self.activation),
      'use_bias': self.use_bias,
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'activity_regularizer':
        regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint),
      'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(SparseWeightNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


