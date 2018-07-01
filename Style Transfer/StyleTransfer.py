#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:49:11 2018

@author: WangJianqiao
"""

import sys
# sys.path.append('./DL Project')
from VGG19 import VGG19
import numpy as np
import tensorflow as tf
from PIL import Image

class StyleTransfer:
    def __init__(self, content_layers, style_layers, content_image,
                 style_image, loss_ratio, num_iter, init_image=None):
        #content_layers and style_layers should be dicts: {name: weight}
        self.net = VGG19('./imagenet-vgg-verydeep-19.mat')
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content = np.float32(content_image)
        self.style = np.float32(style_image)
        self.alpha = loss_ratio
        self.beta = 1
        self.iteration = num_iter
        self.mean = (np.mean(content_image, axis=(0, 1)) + np.mean(style_image, axis=(0, 1))) / (2 * 255)
        
        self.p = tf.constant(content_image - self.mean, name='content')
        self.a = tf.constant(style_image - self.mean, name='style')
        if init_image is None:
            self.img = tf.Variable(tf.random_normal(self.content.shape), trainable=True, dtype=tf.float32)
        else:
            self.img = tf.Variable(init_image - self.mean, trainable=True, dtype=tf.float32)
        
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self._built_net()
    
    def _built_net(self):
        
        #get content layers
        p_vgg = self.net.forward([self.p], scope='content')
        self.p_vgg = {}
        for layer in self.content_layers.keys():
            self.p_vgg[layer] = p_vgg[layer]
        
        #get style layers
        a_vgg = self.net.forward([self.a], scope='style')
        self.a_vgg = {}
        for layer in self.style_layers.keys():
            self.a_vgg[layer] = a_vgg[layer]
        
        #get layers for white noise
        self.x = self.net.forward([self.img], scope='transfered')
        
        #compute content loss
        self.content_loss = 0
        for layer in self.p_vgg.keys():
            x_content = self.x[layer]
            p_content = self.p_vgg[layer]
            content_weight = self.content_layers[layer]
            self.content_loss += content_weight * tf.nn.l2_loss(x_content - p_content) / 2
        
        #compute style loss
        self.style_loss = 0
        for layer in self.a_vgg.keys():
            x_layer = self.x[layer]
            x_style = self._gram_matrix(x_layer)
            a_style = self._gram_matrix(self.a_vgg[layer])
            _, height, width, filters = x_layer.get_shape()
            M = height.value * width.value
            N = filters.value
            style_weight = self.style_layers[layer]
            self.style_loss += style_weight * tf.nn.l2_loss(x_style - a_style) / (4 * N * M)
        
        self.total_loss = self.alpha * self.content_loss + self.beta * self.style_loss
        
    def _gram_matrix(self, tensor):
        shape = tensor.get_shape()
        channel = int(shape[3])
        matrix = tf.reshape(tensor, shape=[-1, channel])
        return tf.matmul(tf.transpose(matrix), matrix)
    
    def update(self, learning_rate, decay_step, decay_rate):
        
        global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.train.exponential_decay(learning_rate, 
                                                        global_step=global_step,
                                                        decay_steps=decay_step,
                                                        decay_rate=decay_rate)

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        add_global = global_step.assign_add(1)
        
        # with tf.Session() as sess:
        # self.sess.run(tf.global_variables_initializer())
        for i in range(self.iteration):
            _, g, loss, target_image = self.sess.run([optimizer, add_global, self.total_loss, self.img])
            if i % 100 == 0:
                print('iteration:', i, 'loss:', loss)
                image = np.clip(target_image + self.mean, 0, 255).astype(np.uint8)
                img = Image.fromarray(image)
                img.save('./output/%d.jpg' % i)
        

        
        