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
"""
  Test Tensorflow version of SparseMNISTNet model
"""
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import timeline

from htmresearch.frameworks import tensorflow as htm
from htmresearch.frameworks.tensorflow.layers.kwinner_layer import (
  _compute_kwinners)

# Parameters from the ExperimentQuick
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.04
MOMENTUM = 0.0
BOOST_STRENGTH = 1.0
BOOST_STRENGTH_FACTOR = 0.9
SEED = 42
N = 500
K = 50
WEIGHT_SPARSITY = 0.4
K_INFERENCE_FACTOR = 2.0
INPUT_SIZE = 28 * 28

OPTIMIZER = "Adam"
LOSS = "sparse_categorical_crossentropy"

# Tensorflow configuration.
# Make sure to use one thread in order to keep the results deterministic
CONFIG = tf.ConfigProto(
  # intra_op_parallelism_threads=8,
  # inter_op_parallelism_threads=8,
  # device_count={'CPU': 8}
)



# import os
# os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0 "


class SparseMNISTNetTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tf.set_random_seed(SEED)

    # Load MNIST dataset into tensors
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    cls.x_train = x_train.reshape(-1, INPUT_SIZE) / 255.0
    cls.x_test = x_test.reshape(-1, INPUT_SIZE) / 255.0
    cls.y_train = y_train
    cls.y_test = y_test


  import unittest
  @unittest.skip("DEBUG: Enable if you need to collect baseline logs for tensorboard")
  def testMNISTBaseline(self):
    # Collect tensorboard baseline logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testMNISTBaseline/{}".format(datetime.now()),
      write_graph=True,
      write_images=True)]

    # Create Simple Dense NN as baseline
    model = keras.Sequential([
      keras.layers.Dense(N, activation=tf.nn.relu, name="l1"),
      keras.layers.Dense(10, activation=tf.nn.softmax, name="l2")
    ])

    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=['accuracy'])

    with self.test_session(config=CONFIG):
      # train
      model.fit(self.x_train, self.y_train,
                epochs=EPOCHS,
                verbose=1,
                batch_size=BATCH_SIZE,
                callbacks=callbacks)
      # test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)

      print 'Test accuracy:', accuracy, "Test loss:", loss
      self.assertAlmostEqual(accuracy, 0.9693, places=4)
      self.assertAlmostEqual(loss, 0.1037, places=4)


  @unittest.skip("")
  def testSparseConstraint(self):
    expected = [float(round(N * WEIGHT_SPARSITY))] * BATCH_SIZE
    constraint = htm.constraints.Sparse(sparsity=WEIGHT_SPARSITY)
    with self.test_session(config=CONFIG):
      actual = constraint(tf.ones([BATCH_SIZE, N]))
      tf.global_variables_initializer().run()
      self.assertAllEqual(tf.count_nonzero(actual, axis=1).eval(), expected)


  @unittest.skip("")
  def testComputeKwinners(self):
    x = np.float32(np.random.uniform(size=(BATCH_SIZE, N)))
    dutyCycles = np.random.uniform(size=(N,))

    # Compute k-winner using numpy
    density = float(K) / N
    boostFactors = np.exp((density - dutyCycles) * BOOST_STRENGTH)
    boosted = x * boostFactors

    # top k
    indices = np.argsort(-boosted, axis=1)[:, :K]
    expected = np.zeros_like(x)
    for i in xrange(BATCH_SIZE):
      expected[i, indices[i]] = x[i, indices[i]]

    # Compute k-winner using tensorflow
    with self.test_session(config=CONFIG):
      actual = _compute_kwinners(x, K, dutyCycles, BOOST_STRENGTH)
      self.assertAllEqual(actual, expected)


  @unittest.skip("")
  def testSparseMNISTNet(self):
    # Collect tensorboard logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testSparseMNISTNet/{}".format(datetime.now()),
      batch_size=BATCH_SIZE,
      write_graph=True,
      write_grads=True,
      write_images=True)]
    callbacks = None
    print ("=" * 80)
    print("testSparseMNISTNet")
    print ("=" * 80)

    model = htm.SparseMNISTNet(n=N, k=K, kInferenceFactor=K_INFERENCE_FACTOR,
                               weightSparsity=WEIGHT_SPARSITY,
                               boostStrength=BOOST_STRENGTH,
                               boostStrengthFactor=BOOST_STRENGTH_FACTOR)
    # Build
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    with self.test_session(config=CONFIG):
      # Train
      model.fit(self.x_train, self.y_train, callbacks=callbacks,
                epochs=EPOCHS, batch_size=BATCH_SIZE)

      # Test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)
      print 'Test accuracy:', accuracy, "Test loss:", loss

      self.assertAlmostEqual(accuracy, 0.9527, places=4)
      self.assertAlmostEqual(loss, 0.1575, places=4)


  # @unittest.skip("")
  def testSparseWeightMNISTNet(self):
    # Collect tensorboard logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testSparseWeightMNISTNet/{}".format(datetime.now()),
      batch_size=BATCH_SIZE,
      write_graph=True,
      write_grads=True,
      write_images=True)]
    callbacks = None
    print ("=" * 80)
    print("testSparseWeightMNISTNet")
    print ("=" * 80)

    model = htm.SparseWeightMNISTNet(n=N, k=K,
                                     kInferenceFactor=K_INFERENCE_FACTOR,
                                     weightSparsity=WEIGHT_SPARSITY,
                                     boostStrength=BOOST_STRENGTH,
                                     boostStrengthFactor=BOOST_STRENGTH_FACTOR)
    # Build
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    with self.test_session(config=CONFIG):
      # Train
      model.fit(self.x_train, self.y_train, callbacks=callbacks,
                epochs=EPOCHS, batch_size=BATCH_SIZE)

      # Test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)
      print 'Test accuracy:', accuracy, "Test loss:", loss

      self.assertAlmostEqual(accuracy, 0.9527, places=4)
      self.assertAlmostEqual(loss, 0.1575, places=4)


  @unittest.skip("")
  def testSparseMNISTNet_S_S(self):
    # Collect tensorboard logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testSparseMNISTNet_S_S/{}".format(datetime.now()),
      batch_size=BATCH_SIZE,
      write_graph=True,
      write_grads=True,
      write_images=True)]
    callbacks = None
    print ("=" * 80)
    print("testSparseMNISTNet_S_S: Sparse Weight, Sparse Input")
    print ("=" * 80)

    model = htm.SparseMNISTNet_S_S(n=N, k=K, kInferenceFactor=K_INFERENCE_FACTOR,
                                   weightSparsity=WEIGHT_SPARSITY,
                                   boostStrength=BOOST_STRENGTH,
                                   boostStrengthFactor=BOOST_STRENGTH_FACTOR)
    # Build
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    with self.test_session(config=CONFIG):
      # Train
      model.fit(self.x_train, self.y_train, callbacks=callbacks,
                epochs=EPOCHS, batch_size=BATCH_SIZE)

      # Test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)
      print 'Test accuracy:', accuracy, "Test loss:", loss


  @unittest.skip("")
  def testSparseMNISTNet_S_D(self):
    # Collect tensorboard logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testSparseMNISTNet_S_D/{}".format(datetime.now()),
      batch_size=BATCH_SIZE,
      write_graph=True,
      write_grads=True,
      write_images=True)]
    callbacks = None
    print ("=" * 80)
    print("testSparseMNISTNet_S_D: Sparse Weight, Dense Input")
    print ("=" * 80)

    model = htm.SparseMNISTNet_S_D(n=N, k=K, kInferenceFactor=K_INFERENCE_FACTOR,
                                   weightSparsity=WEIGHT_SPARSITY,
                                   boostStrength=BOOST_STRENGTH,
                                   boostStrengthFactor=BOOST_STRENGTH_FACTOR)
    # Build
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    with self.test_session(config=CONFIG):
      # Train
      model.fit(self.x_train, self.y_train, callbacks=callbacks,
                epochs=EPOCHS, batch_size=BATCH_SIZE)

      # Test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)
      print 'Test accuracy:', accuracy, "Test loss:", loss


  @unittest.skip("")
  def testSparseMNISTNet_D_S(self):
    # Collect tensorboard logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testSparseMNISTNet_D_S/{}".format(datetime.now()),
      batch_size=BATCH_SIZE,
      write_graph=True,
      write_grads=True,
      write_images=True)]
    callbacks = None
    print ("=" * 80)
    print("testSparseMNISTNet_D_S: Dense Weight, Sparse Input")
    print ("=" * 80)

    model = htm.SparseMNISTNet_D_S(n=N, k=K, kInferenceFactor=K_INFERENCE_FACTOR,
                                   weightSparsity=WEIGHT_SPARSITY,
                                   boostStrength=BOOST_STRENGTH,
                                   boostStrengthFactor=BOOST_STRENGTH_FACTOR)
    # Build
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    with self.test_session(config=CONFIG):
      # Train
      model.fit(self.x_train, self.y_train, callbacks=callbacks,
                epochs=EPOCHS, batch_size=BATCH_SIZE)

      # Test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)
      print 'Test accuracy:', accuracy, "Test loss:", loss


  @unittest.skip("")
  def testSparseMNISTNet_D_D(self):
    # Collect tensorboard logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testSparseMNISTNet_D_D/{}".format(datetime.now()),
      batch_size=BATCH_SIZE,
      write_graph=True,
      write_grads=True,
      write_images=True)]
    callbacks = None
    print ("=" * 80)
    print("testSparseMNISTNet_D_D: Dense Weight, Dense Input")
    print ("=" * 80)

    model = htm.SparseMNISTNet_D_D(n=N, k=K, kInferenceFactor=K_INFERENCE_FACTOR,
                                   weightSparsity=WEIGHT_SPARSITY,
                                   boostStrength=BOOST_STRENGTH,
                                   boostStrengthFactor=BOOST_STRENGTH_FACTOR)
    # Build
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    with self.test_session(config=CONFIG):
      # Train
      model.fit(self.x_train, self.y_train, callbacks=callbacks,
                epochs=EPOCHS, batch_size=BATCH_SIZE)

      # Test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)
      print 'Test accuracy:', accuracy, "Test loss:", loss



if __name__ == "__main__":
  # tf.test.main()

  tf.set_random_seed(SEED)

  # Collect tensorboard logs
  log_dir = "logs/SparseWeightMNISTNet/{}".format(datetime.now())
  callbacks = [keras.callbacks.TensorBoard(
    log_dir=log_dir,
    batch_size=BATCH_SIZE,
    write_graph=True,
    write_grads=True,
    write_images=True)]

  # Load MNIST dataset into tensors
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  x_train = x_train.reshape(-1, INPUT_SIZE) / 255.0
  x_test = x_test.reshape(-1, INPUT_SIZE) / 255.0

  # Build
  model = htm.SparseMNISTNet(n=N, k=K,
                             kInferenceFactor=K_INFERENCE_FACTOR,
                             weightSparsity=WEIGHT_SPARSITY,
                             boostStrength=BOOST_STRENGTH,
                             boostStrengthFactor=BOOST_STRENGTH_FACTOR)

  # with tf.contrib.tfprof.ProfileContext(log_dir) as pctx:
  # Initial complile to get input shape for run_metadata
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'],
                options=run_options, run_metadata=run_metadata)
  model._function_kwargs = {"options": run_options,
                            "run_metadata": run_metadata}

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(log_dir)

  # Train
  summary = model.fit(x_train, y_train,
                      epochs=EPOCHS,
                      verbose=1,
                      batch_size=BATCH_SIZE,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks)

  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
  with open(log_dir + '/SparseWeightMNISTNet.json', 'w') as f:
    f.write(trace.generate_chrome_trace_format())
  train_writer.add_run_metadata(run_metadata, "train")

  # Test
  loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

  print 'Test accuracy:', accuracy, "Test loss:", loss
