#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:49:01 2018

@author: WangJianqiao
"""

import sys
sys.path.append('./')
import scipy.io
from VGG16 import VGG16
import tensorflow as tf
from PIL import Image
import numpy as np
from ImageTransformation import ImageTransformation
model_path = './model.ckpt'

class FastTransfer:
    def __init__(self, content_layers, style_layers, style_image,
                 lambda_content, lambda_style, lambda_tv, print_loss, path, learning_rate, restore_flag=0):
        #content_layers and style_layers should be dicts: {name: weight}
        self.vgg_path = './imagenet-vgg-verydeep-16.mat'
        self.content_layers = content_layers
        self.style_layers = style_layers
        # self.content = np.float32(content_image)
        # self.style = np.float32(style_image)
        self.lambda_content= lambda_content
        self.lambda_style = lambda_style
        self.lambda_tv = lambda_tv
        self.learning_rate = learning_rate
        
        # self.iteration = num_iter
        # self.mean = (np.mean(content_image, axis=(0, 1)) + np.mean(style_image, axis=(0, 1))) / (2 * 255)
        self.mean = np.mean(style_image, axis=(0, 1))
        # self.content = tf.constant(content_image - self.mean, name='content')
        # self.style = tf.constant(style_image - self.mean, name='style')
        self.style_image = style_image - self.mean
        self.path = path

        # Image Transformation
        # self.transform_net = ImageTransformation()
        # print(tf.constant(content_image - self.mean, dtype='float32'))
        self.shape = style_image.shape
        # self.img = self.transform_net(tf.reshape(tf.constant(content_image - self.mean, dtype='float32'), (1, self.shape[0], self.shape[1], self.shape[2])))

        self.count = 0 # count update
        self.print = print_loss
        
        self.restore_flag = restore_flag
        self.sess = tf.Session()
        
        
        self._built_transformation_net()
        self._built_net()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())
    
    def _VGG16(self, path, image, scope=None):
        data = scipy.io.loadmat(path)
        parameters = data['layers'][0]
        layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'pool5')
        network = {}
        layer = image
        with tf.variable_scope(scope):
            for i, name in enumerate(layers):
                key = name[:4]
                if key == 'conv':
                    weight, bias = parameters[i][0][0][2][0]       
                    weight = np.transpose(weight, (1, 0, 2, 3))
                    conv = tf.nn.conv2d(layer, tf.constant(weight), strides=(1,1,1,1), padding='SAME', name=name)
                    layer = tf.nn.bias_add(conv, bias.reshape(-1))
                elif key == 'relu':
                    layer = tf.nn.relu(layer)
                elif key == 'pool':
                    layer = tf.nn.max_pool(layer, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
                network[name] = layer
        return network

    def _weight_variable(self, shape, stddev=0.1, name=None):
    # initialize weighted variables.
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

    def _conv2d(self, x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
        # set convolution layers.
        assert isinstance(x, tf.Tensor)
        return tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)

    def _batch_norm(self, x):
        assert isinstance(x, tf.Tensor)
        # reduce dimension 1, 2, 3, which would produce batch mean and batch variance.
        mean, var = tf.nn.moments(x, axes=[1, 2, 3])
        return tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-5)

    def _relu(self, x):
        assert isinstance(x, tf.Tensor)
        return tf.nn.relu(x)

    def _deconv2d(self, x, W, strides=[1, 2, 2, 1], p='SAME', name=None):
        assert isinstance(x, tf.Tensor)
        # print(W.get_shape().as_list())
        _, _, c, _ = W.get_shape().as_list()
        b, h, w, _ = x.get_shape().as_list()
        # print(b, h, w, c)

        return tf.nn.conv2d_transpose(x, W, [1, strides[1] * h, strides[2] * w, c], strides=strides, padding=p, name=name)
    
    def _residual_block(self, x, idx, w1, w2, strides=[1, 1, 1, 1]):
        h = self._relu(self._batch_norm(self._conv2d(x, w1, strides, name='R' + str(idx) + '_conv1')))
        h = self._batch_norm(self._conv2d(h, w2, name='R' + str(idx) + '_conv2'))
        return x + h

    def _built_transformation_net(self):

        # convolutional weights
        self.c1 = self._weight_variable([9, 9, 3, 32], name='t_conv1_w')
        self.c2 = self._weight_variable([3, 3, 32, 64], name='t_conv2_w')
        self.c3 = self._weight_variable([3, 3, 64, 128], name='t_conv3_w')

        # residual bloc weights
        self.r1_1 = self._weight_variable([3, 3, 128, 128], name='R1_conv1_w')
        self.r1_2 = self._weight_variable([3, 3, 128, 128], name='R1_conv2_w')

        self.r2_1 = self._weight_variable([3, 3, 128, 128], name='R2_conv1_w')
        self.r2_2 = self._weight_variable([3, 3, 128, 128], name='R2_conv2_w')

        self.r3_1 = self._weight_variable([3, 3, 128, 128], name='R3_conv1_w')
        self.r3_2 = self._weight_variable([3, 3, 128, 128], name='R3_conv2_w')

        self.r4_1 = self._weight_variable([3, 3, 128, 128], name='R4_conv1_w')
        self.r4_2 = self._weight_variable([3, 3, 128, 128], name='R4_conv2_w')

        self.r5_1 = self._weight_variable([3, 3, 128, 128], name='R5_conv1_w')
        self.r5_2 = self._weight_variable([3, 3, 128, 128], name='R5_conv2_w')

        # de-convolutional weights
        self.d1 = self._weight_variable([3, 3, 64, 128], name='t_dconv1_w')
        self.d2 = self._weight_variable([3, 3, 32, 64], name='t_dconv2_w')
        self.d3 = self._weight_variable([9, 9, 3, 32], name='t_dconv3_w')
        
    def _built_net(self):
        
        # get content layers
        self.content_image = tf.placeholder(tf.float32, [None, self.shape[0], self.shape[1], self.shape[2]], name='content_image')
        self.content = self.transform_net(self.content_image)
        content_vgg = self._VGG16(self.vgg_path, self.content, scope='content')
        self.content_vgg = {}
        for layer in self.content_layers.keys():
            self.content_vgg[layer] = content_vgg[layer]
        
        # get style layers
        self.style = tf.placeholder(tf.float32, [None, self.shape[0], self.shape[1], self.shape[2]], name='style_image')
        style_vgg = self._VGG16(self.vgg_path, self.style, scope='style')
        self.style_vgg = {}
        for layer in self.style_layers.keys():
            self.style_vgg[layer] = style_vgg[layer]
        
        # get layers for init_image
        self.img = tf.placeholder(tf.float32, [None, self.shape[0], self.shape[1], self.shape[2]], name='tansfered_image')
        self.img_vgg = self._VGG16(self.vgg_path, self.img, scope='transfered')
        
        # compute content loss
        self.content_loss = 0
        for layer in self.content_vgg.keys():
            x_content = self.img_vgg[layer]
            p_content = self.content_vgg[layer]
            content_weight = self.content_layers[layer]
            _, height, width, filters = x_content.get_shape()
            M = height.value * width.value
            N = filters.value
            self.content_loss += content_weight * tf.nn.l2_loss(x_content - p_content) / (M * N)
        
        # compute style loss
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
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
    
    def _gram_matrix(self, tensor):
        shape = tensor.get_shape()
        channel = int(shape[3])
        matrix = tf.reshape(tensor, shape=[-1, channel])
        return tf.matmul(tf.transpose(matrix), matrix)

    def transform_net(self, x):
        # convolution
        self.x1 = self._relu(self._batch_norm(self._conv2d(x, self.c1, name='t_conv1')))
        self.x2 = self._relu(self._batch_norm(self._conv2d(self.x1, self.c2, strides=[1, 2, 2, 1], name='t_conv2')))
        self.x3 = self._relu(self._batch_norm(self._conv2d(self.x2, self.c3, strides=[1, 2, 2, 1], name='t_conv3')))
        # print(self.x3)
        # residual block
        self.r1 = self._residual_block(self.x3, 1, self.r1_1, self.r1_2, strides=[1, 1, 1, 1])
        self.r2 = self._residual_block(self.r1, 2, self.r2_1, self.r2_2, strides=[1, 1, 1, 1])
        self.r3 = self._residual_block(self.r2, 3, self.r3_1, self.r3_2, strides=[1, 1, 1, 1])
        self.r4 = self._residual_block(self.r3, 4, self.r4_1, self.r4_2, strides=[1, 1, 1, 1])
        self.r5 = self._residual_block(self.r4, 5, self.r5_1, self.r5_2, strides=[1, 1, 1, 1])
        # de-convolution
        # print(self.r5)
        # print(self.d1)
        self.y1 = self._relu(self._batch_norm(self._deconv2d(self.r5, self.d1, strides=[1, 2, 2, 1], name='t_deconv1')))
        # print(self.y1)
        self.y2 = self._relu(self._batch_norm(self._deconv2d(self.y1, self.d2, strides=[1, 2, 2, 1], name='t_deconv2')))
        self.y3 = self._relu(self._batch_norm(self._deconv2d(self.y2, self.d3, strides=[1, 1, 1, 1], name='t_deconv3')))
        # print(self.y3)
        return tf.multiply((tf.tanh(self.y3) + 1), tf.constant(127.5, tf.float32, shape=self.y3.get_shape()), name='output')
    
    def save(self):
        self.saver.save(self.sess, self.path)
    
    def update(self, content_image):
        if self.restore_flag == 1:
            self.saver.restore(self.sess, self.path)
            self.restore_flag = 0
        # self.img = self.transform_net(tf.reshape(tf.constant(content_image - self.mean, dtype='float32'), (1, self.shape[0], self.shape[1], self.shape[2])))
        # self.content = tf.constant(content_image - self.mean, name='content')
        # self._built_net()


        # self.learning_rate = learning_rate

        # optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        # with tf.Session() as sess:
        # self.sess.run(tf.global_variables_initializer())
            # for i in range(self.iteration):
        _, loss= self.sess.run([self.optimizer, self.total_loss], feed_dict={self.content_image: [content_image - self.mean], self.img: [content_image], self.style: [self.style_image]})
        if self.count % self.print == 0:
            self.save()
            print('loss:', loss)

                # image = np.clip(target_image + self.mean, 0, 255).astype('uint8')
                # image = image.reshape(self.shape)
                # img = Image.fromarray(image)
                # img.save('./fast_output/%d.jpg' % self.count)
        self.count += 1
                # if i % 100 == 0:
                    # print('iteration:', i, 'loss:', loss)
                    # image = np.clip(target_image + self.mean, 0, 255).astype('uint8')
                    # image = image.reshape(self.shape)
                    # img = Image.fromarray(image)
                    # img.save('./fast_output/%d.jpg' % i)
