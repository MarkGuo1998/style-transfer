#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:49:01 2018

@author: WangJianqiao
"""

import sys
sys.path.append('./')
from VGG16 import VGG16
import tensorflow as tf
from PIL import Image
import numpy as np
from net import FastStyleNet

class FastTransfer:
    def __init__(self, content_layers, style_layers, content_image,
                 style_image, lambda_content, lambda_style, lambda_tv, num_iter):
        #content_layers and style_layers should be dicts: {name: weight}
        self.net = VGG16('./imagenet-vgg-verydeep-16.mat')
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content = np.float32(content_image)
        self.style = np.float32(style_image)
        self.lambda_content= lambda_content
        self.lambda_style = lambda_style
        self.lambda_tv = lambda_tv
        self.iteration = num_iter
        self.mean = (np.mean(content_image, axis=(0, 1)) + np.mean(style_image, axis=(0, 1))) / (2 * 255)
        self.content = tf.constant(content_image - self.mean, name='content')
        self.style = tf.constant(style_image - self.mean, name='style')

        # Image Transformation
        self.transform_net = FastStyleNet()
        # print(tf.constant(content_image - self.mean, dtype='float32'))
        self.shape = content_image.shape
        self.img = self.transform_net(tf.reshape(tf.constant(content_image - self.mean, dtype='float32'), (1, self.shape[0], self.shape[1], self.shape[2])))

        
        
        self._built_net()
    
    def _built_net(self):
        
        # get content layers
        content_vgg = self.net.forward([self.content], scope='content')
        self.content_vgg = {}
        for layer in self.content_layers.keys():
            self.content_vgg[layer] = content_vgg[layer]
        
        # get style layers
        style_vgg = self.net.forward([self.style], scope='style')
        self.style_vgg = {}
        for layer in self.style_layers.keys():
            self.style_vgg[layer] = style_vgg[layer]
        
        # get layers for init_image
        self.img_vgg = self.net.forward(self.img, scope='transfered')
        
        #compute content loss
        self.content_loss = 0
        for layer in self.content_vgg.keys():
            x_content = self.img_vgg[layer]
            p_content = self.content_vgg[layer]
            content_weight = self.content_layers[layer]
            _, height, width, filters = x_content.get_shape()
            M = height.value * width.value
            N = filters.value
            self.content_loss += content_weight * tf.nn.l2_loss(x_content - p_content) / (M * N)
        
        #compute style loss
        self.style_loss = 0
        for layer in self.style_vgg.keys():
            x_layer = self.img_vgg[layer]
            x_style = self._gram_matrix(x_layer)
            a_style = self._gram_matrix(self.style_vgg[layer])
            _, height, width, filters = x_layer.get_shape()
            M = height.value * width.value
            N = filters.value
            style_weight = self.style_layers[layer]
            self.style_loss += style_weight * tf.nn.l2_loss(x_style - a_style) / (N * M)
        
        # total variation regularizer
        self.total_variation = tf.reduce_sum(tf.image.total_variation(self.img))
        
        # total loss
        self.total_loss = self.lambda_content * self.content_loss + self.lambda_style * self.style_loss + self.lambda_tv * self.total_variation

    def _gram_matrix(self, tensor):
        shape = tensor.get_shape()
        channel = int(shape[3])
        matrix = tf.reshape(tensor, shape=[-1, channel])
        return tf.matmul(tf.transpose(matrix), matrix)
    
    def update(self, learning_rate):
        

        self.learning_rate = learning_rate

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iteration):
                _, loss, target_image = sess.run([optimizer, self.total_loss, self.img])
                if i % 100 == 0:
                    print('iteration:', i, 'loss:', loss)
                    image = np.clip(target_image + self.mean, 0, 255).astype('uint8')
                    image = image.reshape(self.shape)
                    img = Image.fromarray(image)
                    img.save('./fast_output/%d.jpg' % i)