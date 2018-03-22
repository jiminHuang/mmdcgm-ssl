#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ssl/task.py
# Author: Jimin Huang <huangjimin@whu.edu.cn>
# Date: 21.03.2018
import arrow
import os
import tensorflow as tf

from tensorflow.contrib.learn import learn_runner

from model import model_fn


FLAGS = tf.app.flags.FLAGS


flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/', 'Directory for datum')
flags.DEFINE_integer('buffer_size', 2000, 'Buffer size of shuffle')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('sample_times', 599, 'Batch size')
flags.DEFINE_integer('feature_columns', 784, 'Volumes of feature vectors')


def tf_example(labelled):
    example = {
        'x': tf.FixedLenFeature((FLAGS.feature_columns), tf.float32),
    }
    if labelled:
        example['y'] = tf.FixedLenFeature((), tf.float32)
    return example


def parse_labelled_X(proto):
    parsed_features = tf.parse_single_example(proto, tf_example(True))
    features = {
        'x_labelled': parsed_features['x']
    }
    return features, parsed_features['y']


def parse_unlabelled_X(proto):
    parsed_features = tf.parse_single_example(proto, tf_example(False))
    features = {
        'x_unlabelled': parsed_features['x']
    }
    return features


def get_train_inputs(batch_size, dataset):
    """Returns the input function to get data.

    Args:
        batch_size: (int) Batch size of iterator
        dataset: (str) The name of dataset

    Return:
        Input function
    """
    def func():
        with tf.name_scope(dataset+'_train'):
            label_dataset = tf.data.TFRecordDataset([
                os.path.join(
                    FLAGS.data_dir, dataset, 'train_labelled',
                    'data.tfrecords'
                )
            ]).map(parse_labelled_X).shuffle(FLAGS.buffer_size).repeat(
                FLAGS.iterations
            ).batch(FLAGS.batch_size / FLAGS.sample_times)

            unlabel_dataset = tf.data.TFRecordDataset([
                os.path.join(
                    FLAGS.data_dir, dataset, 'train_unlabelled',
                    'data.tfrecords'
                )
            ]).map(parse_labelled_X).shuffle(FLAGS.buffer_size).repeat(
                FLAGS.iterations
            ).batch(FLAGS.batch_size)

            label_iterator = label_dataset.make_one_shot_iterator()
            unlabel_iterator = unlabel_dataset.make_one_shot_iterator()

            x_labelled, y = label_iterator.get_next()
            x_unlabelled = unlabel_iterator.get_next()
            x = tf.concat([x_labelled, x_unlabelled], 0)

            return x, y
    return func


def get_test_inputs(batch_size, dataset):
    """Returns the input function to get data.

    Args:
        batch_size: (int) Batch size of iterator
        dataset: (str) The name of dataset

    Return:
        Input function
    """
    def func():
        with tf.name_scope(dataset+'_test'):
            label_dataset = tf.data.TFRecordDataset([
                os.path.join(
                    FLAGS.data_dir, dataset, 'test', 'data.tfrecords'
                )
            ]).map(parse_labelled_X).shuffle(
                FLAGS.buffer_size
            ).batch(FLAGS.batch_size)
            label_iterator = label_dataset.make_one_shot_iterator()

            return label_iterator.get_next()
    return func


def run_experiment(argv=None):
    """Run training experiments
    """
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        min_eval_frequency=FLAGS.eval_freq,
        feature_columns=[
            tf.feature_column.numeric_column(
                key='x', shape=(FLAGS.feature_columns,), dtype=tf.float32
            )
        ]
    )

    run_config = tf.contrib.learn.RunConfig(
        save_summary_steps=10,
    )
    run_config = run_config.replace(
        model_dir=os.path.join(FLAGS.model_dir, arrow.now().format())
    )

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )


def experiment_fn(run_config, params):
    """Create an experiment to train and evaluate the model

    Args:
        run_config: Configuration for Estimator run
        params: Hyper parameters

    Return:
        Experiment for training the model
    """
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency
    )

    estimator = get_estimator(run_config, params)

    train_input_fn = get_train_inputs(FLAGS.batch_size, FLAGS.dataset)
    eval_input_fn = get_test_inputs(FLAGS.batch_size, FLAGS.dataset)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=None,
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        eval_steps=None  # Use evaluation feeder until its empty
    )
    return experiment


def get_estimator(run_config, params):
    """Return the model as a (Estimator)

    Args:
        run_config: Configuration for extimator run
        params: hyper parameters
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params
    )


if __name__ == '__main__':
    tf.app.run(main=run_experiment)
