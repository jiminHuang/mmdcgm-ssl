#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ssl/model.py
# Author: Jimin Huang <huangjimin@whu.edu.cn>
# Date: 22.03.2018
import tensorflow as tf

from layer import conv_layer, mlp_layer


def model_fn(features, labels, mode, params):
    """Define the model structure

    Args:
        features: (Dict) of features
        labels: (batch_labels) from input_fn
        mode: (tf.estimator.ModeKeys) an instance
        params: (Hparams) Additional configuration
    """
    features = tf.feature_column.input_layer(
        features, params['feature_columns']
    )
    x = tf.reshape(
        features['x'],
        [-1, params['height'], params['weight'], params['channel']],
    )

    # x2y
    inputs = x
    for index in range(params['nlayers_cla']):
        inputs = conv_layer(
            inputs, params['nk_cla'][index], params['dk_cla'][index],
            params['str_cla'][index], params['pad_cla'][index], tf.nn.relu,
            params['bn_cla'], params['dr_cla'][index], params['ps_cla'][index],
            'Cla_'+str(index+1)
        )

    if params['top_mlp']:
        inputs = tf.layers.flatten(inputs)
        feature = mlp_layer(
            inputs, params['mlp_size'], tf.nn.relu, params['bn_cla'], 0.5,
            name='MLP_Cla'
        )
    else:
        feature = tf.reduce_sum(inputs, [1])

    print feature
