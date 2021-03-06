"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Train", type=float, help="Num Training Steps")
parser.add_argument("Step", type=float, help="Step Size")
parser.add_argument("Reg", type=float, help="Regularization Weight")
parser.add_argument("Eps", type=float, help="Epsilon Perturbation")

args = parser.parse_args()
relu_step = args.Step
lambda_reg = args.Reg
num_itrs = int(args.Train)

print(" ~~~~~~~ ~~~~ ~~~  !! Step:", relu_step, "  Regularization Lambda:", lambda_reg, "  Training Itrs:", num_itrs)

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(MULT_A = relu_step, LAM_REG_WEIGHT =lambda_reg )

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.my_loss,
                                                   global_step=global_step)

# Set up adversary
adv_epsilon = config['epsilon']
adv_epsilon = args.Eps
attack = LinfPGDAttack(model,
                       adv_epsilon,
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
# Changed adv to nat
tf.summary.scalar('accuracy nat train', model.accuracy)
tf.summary.scalar('accuracy nat', model.accuracy)
tf.summary.scalar('xent nat train', model.xent / batch_size)
tf.summary.scalar('xent nat', model.xent / batch_size)
tf.summary.image('images nat train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  max_num_training_steps = num_itrs
  for ii in range(max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size)

    # Compute Adversarial Perturbations
    start = timer()
    #x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    #adv_dict = {model.x_input: x_batch_adv,
    #            model.y_input: y_batch}

    # Output to stdout
    if (ii + 1) % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      #adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Bristy :D Step {}:    ({})'.format(ii, datetime.now()))
      print('    Training nat accuracy {:.4}%'.format(nat_acc * 100.0))
      #print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=nat_dict)  #This was adv_dict
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start
