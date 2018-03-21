#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ssl/preprocess.py
# Author: Jimin Huang <huangjimin@whu.edu.cn>
# Date: 21.03.2018
import numpy as np
import os
import tensorflow as tf

from numpy import random


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset', 'mnist', 'the dataset name'
)
tf.app.flags.DEFINE_string(
    'data_dir', 'data', 'the directory for data'
)


def load_data(dataset):
    """Load dataset from given dataset name

    Args:
        dataset: (str) the name of the dataset

    Return:
        (tuple) of X_train, Y_train, X_test, Y_test
    """
    return tf.keras.datasets.mnist.load_data()


def sample_labelled(train, train_label, size, classes):
    """Sample labeled datum from train

    Args:
        train: (numpy.darray) the datum
        train_label: (numpy.darray) 1d labels
        size: (int) the sample size
        classes: (int) the number of classes

    Return:
        (tuple) of X_labelled, Y_labelled, X_unlabelled
    """
    label_size = size / classes

    random_index = range(train.shape[0])
    random.shuffle(random_index)

    train, train_label = train[random_index], train_label[random_index]

    labelled_indexes = []
    label_indexes = {}
    for index, label in enumerate(train_label):
        label_indexes.setdefault(label, [])
        label_indexes[label].append(index)

    for label, indexes in label_indexes.items():
        labelled_indexes += indexes[:label_size]

    unlabelled_indexes = list(set(random_index) - set(labelled_indexes))

    X_labelled = train[labelled_indexes]
    Y_labelled = train_label[labelled_indexes]
    X_unlabelled = train[unlabelled_indexes]
    Y_unlabelled = train_label[unlabelled_indexes]
    return X_labelled, Y_labelled, X_unlabelled, Y_unlabelled


def _float_feature(value):
    """Generate float feature according to the value

    Args:
        value: a list of float

    Return:
        a `tf.train.Feature`
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def generate_tf_example(args, label):
    """Generate tf example with given datum

    Args:
        args: a list of float
        label: (bool) whether there is y

    Return:
        a `tf.train.Example`
    """
    args = np.array(args)
    feature_dict = {
        'x': _float_feature(args[:-1 if label else len(args)]),
    }
    if label:
        feature_dict['y'] = _float_feature(args[-1])
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def generate_tf_records(dataset, datum, label):
    """Generate tfrecords

    Args:
        dataset: (str) the name to save records
        datum: (numpy.darray) the data to save
        label: (bool) whether datum with label
    """
    try:
        os.mkdir(dataset)
    except OSError:
        tf.logging.info('{} already exists'.format(dataset))

    filename = os.path.join(dataset, ('data.tfrecords'))

    with tf.python_io.TFRecordWriter(filename) as writer:
        for data in datum:
            example = generate_tf_example(data, label)
            writer.write(example.SerializeToString())


def transform_x(data):
    """Transform X to flattern vector
    """
    return data.reshape([data.shape[0], data.shape[1] * data.shape[2]])


def main(_):
    data_dir = FLAGS.data_dir
    train, test = load_data(FLAGS.dataset)
    try:
        os.mkdir(os.path.join(data_dir, FLAGS.dataset))
    except OSError:
        tf.logging.info('{} already exists'.format(FLAGS.dataset))

    X_labelled, Y_labelled, X_unlabelled, _, = sample_labelled(
        train[0], train[1], 100, 10
    )
    generate_tf_records(
        os.path.join(data_dir, FLAGS.dataset, 'train_unlabelled'),
        transform_x(X_unlabelled), False
    )

    label_datum = np.hstack([transform_x(X_labelled), np.matrix(Y_labelled).T])
    test_datum = np.hstack([transform_x(test[0]), np.matrix(test[1]).T])

    generate_tf_records(
        os.path.join(data_dir, FLAGS.dataset, 'train_labelled'), label_datum,
        True
    )

    generate_tf_records(
        os.path.join(data_dir, FLAGS.dataset, 'test'), test_datum, True
    )


main(None)
