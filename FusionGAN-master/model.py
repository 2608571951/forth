# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

train_MRT1_path = 'dataset_images/Train_MRT1'
train_MRT2_path = 'dataset_images/Train_MRT2'
test_MRT1_path = 'dataset_images/Test_MRT1'
test_MRT2_path = 'dataset_images/Test_MRT2'


class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=256,
               label_size=256,
               batch_size=2,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('MRT2_input'):
        #红外图像patch
        self.images_MRT2 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_MRT2')
        self.labels_MRT2 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_MRT2')
    with tf.name_scope('MRT1_input'):
        #可见光图像patch
        self.images_MRT1 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_MRT1')
        self.labels_MRT1 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_MRT1')
        #self.labels_vi_gradient=gradient(self.labels_vi)
    #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
        self.input_image=tf.concat([self.images_MRT1, self.images_MRT2], axis=-1)
    #self.pred=tf.clip_by_value(tf.sign(self.pred_ir-self.pred_vi),0,1)
    #融合图像
    with tf.name_scope('fusion'):
        print('start fusion_model......')
        print('input_image.shape: ', self.input_image.shape)
        self.fusion_image = self.fusion_model(self.input_image)
    with tf.name_scope('d_loss'):
        #判决器对可见光图像和融合图像的预测
        #pos=self.discriminator(self.labels_vi,reuse=False)
        pos=self.discriminator(self.labels_MRT1, reuse=False)
        print('pos={}'.format(pos))
        neg=self.discriminator(self.fusion_image, reuse=True, update_collection='NO_OPS')
        print('neg={}'.format(neg))
        #把真实样本尽量判成1否则有损失（判决器的损失）
        #pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos, labels=tf.ones_like(pos)))
        #pos_loss=tf.reduce_mean(tf.square(pos-tf.ones_like(pos)))
        pos_loss=tf.reduce_mean(tf.square(pos-tf.random_uniform(shape=[self.batch_size, 1], minval=0.7, maxval=1.2, dtype=tf.float32)))
        #把生成样本尽量判断成0否则有损失（判决器的损失）
        #neg_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.zeros_like(neg)))
        #neg_loss=tf.reduce_mean(tf.square(neg-tf.zeros_like(neg)))
        neg_loss=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size, 1], minval=0, maxval=0.3, dtype=tf.float32)))
        self.d_loss=neg_loss + pos_loss
        tf.summary.scalar('loss_d',self.d_loss)
    with tf.name_scope('g_loss'):
        #self.g_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.ones_like(pos)))
        self.g_loss_1 = tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size, 1], minval=0.7, maxval=1.2, dtype=tf.float32)))
        tf.summary.scalar('g_loss_1', self.g_loss_1)
        #self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))
        self.g_loss_2 = 2. * tf.reduce_mean(tf.square(self.fusion_image - self.labels_MRT2)) + \
                            2. * tf.reduce_mean(tf.square(self.fusion_image - self.labels_MRT1)) + \
                        4. * tf.reduce_mean(tf.square(gradient(self.fusion_image) - gradient (self.labels_MRT1)))
        tf.summary.scalar('g_loss_2', self.g_loss_2)
        self.g_loss_total = 1. * self.g_loss_1 + 5. * self.g_loss_2
        tf.summary.scalar('loss_g', self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=50)


  def create_data_h5(self, config):
      savepath_Train_MRT1 = input_setup(self.sess, config, train_MRT1_path)
      savepath_Train_MRT2 = input_setup(self.sess, config, train_MRT2_path)
      return savepath_Train_MRT1, savepath_Train_MRT2


  def train(self, config):
    if not os.path.exists(os.path.join('.', 'dataset_h5')):
      data_dir_MRT1, data_dir_MRT2 = self.create_data_h5(config)
      print('创建MRT1训练数据的h5文件：Train_MRT1_dir = {}'.format(data_dir_MRT1))
      print('创建MRT2训练数据的h5文件：Train_MRT2_dir = {}'.format(data_dir_MRT2))
    else:
      data_dir_MRT1 = './dataset_h5/Train_MRT1/train.h5'
      data_dir_MRT2 = './dataset_h5/Train_MRT2/train.h5'

    train_data_MRT1, train_label_MRT1 = read_data(data_dir_MRT1) # 读取h5文件中的数据
    train_data_MRT2, train_label_MRT2 = read_data(data_dir_MRT2)

    #找训练时更新的变量组（判决器和生成器是分开训练的，所以要找到对应的变量）
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    print("判别器变量组：{}".format(self.d_vars))
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print("生成器变量组：{}".format(self.g_vars))

    # Stochastic gradient descent with the standard backpropagation
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total, var_list=self.g_vars)
        self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)

    #将所有统计的量合起来
    self.summary_op = tf.summary.merge_all()
    #生成日志文件
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train', self.sess.graph, flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_MRT1) // config.batch_size
        print('len(train_data_MRT1)={};; config.batch_size={};; batch_num={}'.format(len(train_data_MRT1), config.batch_size,batch_idxs))
        for idx in range(0, batch_idxs):
          batch_images_MRT1 = train_data_MRT1[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_MRT1 = train_label_MRT1[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_MRT2 = train_data_MRT2[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_MRT2 = train_label_MRT2[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          for i in range(2):
            _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_MRT1: batch_images_MRT1,
                                    self.images_MRT2: batch_images_MRT2, self.labels_MRT1: batch_labels_MRT1, self.labels_MRT2: batch_labels_MRT2})
            # print("neg.eval()={}".format(self.sess.run(neg)))
            # self.sess.run(self.clip_disc_weights)
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total, self.summary_op], feed_dict={self.images_MRT1: batch_images_MRT1,
                                    self.images_MRT2: batch_images_MRT2, self.labels_MRT1: batch_labels_MRT1, self.labels_MRT2: batch_labels_MRT2})
          #将统计的量写到日志文件里
          self.train_writer.add_summary(summary_str, counter)

          if counter % 20 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_d, err_g))
          if counter % 500 == 0:
            self.save(config.checkpoint_dir, ep)



  def fusion_model(self,img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",[3,3,2,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1",[128],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            print('conv1.shape: ', conv1.shape)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",[3,3,128,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2",[64],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
            print('conv2.shape: ', conv2.shape)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3",[32],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
            print('conv3.shape: ', conv3.shape)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4",[16],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
            print('conv4.shape: ', conv4.shape)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[1,1,16,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
            conv5= tf.nn.conv2d(conv4, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5=tf.nn.tanh(conv5)
            print('conv5.shape: ', conv5.shape)
    print('end fusion_model')
    return conv5
    
  def discriminator(self, img, reuse, update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        print('img.shape={}'.format(img.shape))
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_1",[16],initializer=tf.constant_initializer(0.0))
            conv1=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1 = lrelu(conv1)
            print('conv11.shape: ', conv1.shape)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",[3,3,16,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_2",[32],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,2,2,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
            print('conv12.shape: ', conv2.shape)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_3",[64],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,2,2,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3=lrelu(conv3)
            print('conv13.shape: ', conv3.shape)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_4",[128],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3, weights, strides=[1,2,2,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4=lrelu(conv4)
            print('conv14.shape: ', conv4.shape)
            conv4 = tf.reshape(conv4, [self.batch_size, int(conv4.shape[1]) * int(conv4.shape[2]) * int(conv4.shape[3])])
            print('conv14_2.shape: ', conv4.shape)
        with tf.variable_scope('line_5'):
            print(int(conv4.shape[1]))
            weights=tf.get_variable("w_5", [int(conv4.shape[1]), 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            # print(weights.shape)
            weights=weights_spectral_norm(weights, update_collection=update_collection)
            bias=tf.get_variable("b_5", [1], initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4, weights) + bias
    print("line_5={}".format(line_5))
            #conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    return line_5

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
