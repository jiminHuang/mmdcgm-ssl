#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ssl/layer.py
# Author: Jimin Huang <huangjimin@whu.edu.cn>
# Date: 22.03.2018
import tensorflow as tf


def conv_layer(
    inputs, filters, kernel_size, strides, padding, activation, batch_norm,
    dropout, max_pool, name
):
    """A compositional layer of Conv2d + BatchNorm + Pooling + Dropout

    Args:
        inputs: (Tensor) input
        filters: (int) the number of filters in the convolution.
        kernel_size: (int / iterator of 2 int) the height and width of the 2D
        convolution filter. Can be a **SINGLE** int to specify the same number.
        strides: (int / iterator of 2 int) the strides of the convolution along
        the height and width.
        padding: (str) "valid" or "same".
        activation: (Callable / None) activation function.
        batch_norm: (bool) Whether a batch_norm layer is added
        dropout: (int) if greater than 0, then a dropout layer with it is
        added.
        max_pool: (int) if greater than 1, then a max_pooling layer with pool
        size of (max_pool, max_pool) is added.
        name: (str) the name of the layer

    Return:
        (Tensor) output
    """
    with tf.name_scope(name):
        output = tf.layers.conv2d(
            inputs, filters, kernel_size, strides, padding,
            activation=activation, name='Conv_'+name
        )
        if batch_norm:
            output = tf.layers.batch_normalization(
                output, name='Batch_'+name
            )

        if max_pool > 1:
            output = tf.layers.max_pooling2d(
                output, max_pool, name='MaxPool_'+name
            )

        if dropout > 0:
            output = tf.layers.dropout(
                output, dropout, name='Dropout_'+name
            )
        return output


def mlp_layer(inputs, units, activation, batch_norm, dropout, name):
    """A composite layer of Dense + BatchNorm + Dropout

    Args:
        inputs: (Tensor) input.
        units: (int) dimensionality of the output space.
        activation: (Callable / None) activation function.
        batch_norm: (bool) Whether a batch norm layer is added.
        dropout: (float) if greater than 0, a dropout layer is added.
        name: (str) the name of layer.

    Return:
        (Tensor) output
    """
    output = tf.layers.dense(
        inputs, units, activation=activation, name='MLP_'+name
    )

    if batch_norm:
        output = tf.layers.batch_normalization(output, name='Batch_'+name)

    if dropout > 0:
        output = tf.layers.dropout(
            output, dropout, name='Dropout_'+name
        )

    return output
