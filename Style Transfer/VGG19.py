#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 09:10:01 2018

@author: WangJianqiao
"""

import scipy.io
import numpy as np
import tensorflow as tf

data = scipy.io.loadmat('./imagenet-vgg-verydeep-19.mat')
weight = data['layers'][0]

class VGG19:
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        self.parameters = data['layers'][0]
        self.layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4')
    
    def forward(self, image, scope=None):
        self.network = {}
        layer = image
        with tf.variable_scope(scope):
            for i, name in enumerate(self.layers):
                key = name[:4]
                if key == 'conv':
                    weight, bias = self.parameters[i][0][0][0][0]       
                    weight = np.transpose(weight, (1, 0, 2, 3))
                    conv = tf.nn.conv2d(layer, tf.constant(weight), strides=(1,1,1,1), padding='SAME', name=name)
                    layer = tf.nn.bias_add(conv, bias.reshape(-1))
                    layer = tf.nn.relu(layer)
                elif key == 'pool':
                    layer = tf.nn.max_pool(layer, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
                self.network[name] = layer
        return self.network

'''
vgg = VGG19('imagenet-vgg-verydeep-19.mat')
from PIL import Image
img = Image.open('starry-night.jpg')
img = np.array(img).astype('float32')
img = tf.constant(img)
network = vgg.forward([img])
'''