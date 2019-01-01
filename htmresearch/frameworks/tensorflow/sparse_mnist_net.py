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
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

from htmresearch.frameworks.tensorflow import (constraints, layers)



def SparseMNISTNet(n=2000, k=200, kInferenceFactor=1.0, weightSparsity=0.5,
                   boostStrength=1.0, boostStrengthFactor=1.0, name=None):
  """
      A network with one hidden layer, which is a k-sparse linear layer, designed
    for MNIST.

    :param n:
      Number of units in the hidden layer.
    :type n: int

    :param k:
      Number of ON (non-zero) units per iteration.
    :type k: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero.
    :type weightSparsity: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      boost strength is multiplied by this factor after each epoch.
      A value < 1.0 will decrement it every epoch.
    :type boostStrengthFactor: float

    :param name: Optional Model name
    :type name: str

  :return: Configured tf.keras.model designed for MNIST
  :rtype: keras.Model
  """

  constraint = constraints.Sparse(sparsity=weightSparsity,
                                  name="{}_constraint".format(weightSparsity))
  glorot_uniform = keras.initializers.get("glorot_uniform")
  initializer = lambda *args, **kwargs: constraint(glorot_uniform(*args,
                                                                  **kwargs))
  model = keras.Sequential(name=name)

  # Hidden sparse NN layer
  model.add(keras.layers.Dense(name="l1", units=n, activation=tf.nn.relu,
                               kernel_initializer=initializer,
                               kernel_constraint=constraint))
  # K-Winners
  model.add(layers.KWinner(name="kwinner", k=k,
                           kInferenceFactor=kInferenceFactor,
                           boostStrength=boostStrength,
                           boostStrengthFactor=boostStrengthFactor))
  # Output NN layer
  model.add(keras.layers.Dense(name="l2", units=10, activation=tf.nn.softmax))

  return model



def SparseWeightMNISTNet(n=2000, k=200, kInferenceFactor=1.0, weightSparsity=0.5,
                         boostStrength=1.0, boostStrengthFactor=1.0, name=None):
  """
      A network with one hidden layer, which is a k-sparse linear layer, designed
    for MNIST.

    :param n:
      Number of units in the hidden layer.
    :type n: int

    :param k:
      Number of ON (non-zero) units per iteration.
    :type k: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero.
    :type weightSparsity: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      boost strength is multiplied by this factor after each epoch.
      A value < 1.0 will decrement it every epoch.
    :type boostStrengthFactor: float

    :param name: Optional Model name
    :type name: str

  :return: Configured tf.keras.model designed for MNIST
  :rtype: keras.Model
  """

  model = keras.Sequential(name=name or "SparseWeightMNISTNet")

  # Hidden sparse NN layer
  model.add(layers.SparseWeightNN(name="l1", units=n, activation=tf.nn.relu,
                                  sparsity=weightSparsity))
  # K-Winners
  model.add(layers.KWinner(name="kwinner", k=k,
                           kInferenceFactor=kInferenceFactor,
                           boostStrength=boostStrength,
                           boostStrengthFactor=boostStrengthFactor))
  # Output NN layer
  model.add(keras.layers.Dense(name="l2", units=10, activation=tf.nn.softmax))

  return model

