#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ssl/task.py
# Author: Jimin Huang <huangjimin@whu.edu.cn>
# Date: 21.03.2018


train, test = tf.keras.datasets.mnist.load_data()


def get_inputs(batch_size, datum, dataset):
    """Returns the input function to get data.

    Args:
        batch_size: (int) Batch size of iterator
        datum: (Tensor) The input data
        dataset: (str) The name of dataset
    
    Return:
        Input function
    """
    with tf.name_scope(dataset+'_data'):
        
    
