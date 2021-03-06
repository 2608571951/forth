# -*- coding: utf-8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  #读到图片
  image = imread(path, is_grayscale=True)
  #将图片label裁剪为scale的倍数
  label_ = modcrop(image, scale)

  # Must be normalized
  image = (image-127.5 )/ 127.5 
  label_ = (image-127.5 )/ 127.5 
  #下采样之后再插值
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_


def prepare_data(sess, dataset):
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.png"))
    #将图片按序号排序
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data


def make_data(sess, data, label, data_dir):

  dataset_dir = 'dataset_h5'
  str_data_dir = data_dir.split('/')
  # print('str_data_dir[1]: ', str_data_dir[1])

  savepath = os.path.join('.', os.path.join(dataset_dir, str_data_dir[1], 'train.h5'))
  if not os.path.exists(os.path.join('.', os.path.join(dataset_dir, str_data_dir[1]))):
    os.makedirs(os.path.join('.', os.path.join(dataset_dir, str_data_dir[1])))

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)
  return savepath



def imread(path, is_grayscale=True):

  if is_grayscale:
    #flatten=True 以灰度图的形式读取 
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0: h, 0: w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0: h, 0: w]
  return image


def input_setup(sess, config, data_dir, index=0):      # 'dataset_images/Train_MRT1'
  print('start input_setup......')
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if config.is_train:
    #取到所有的原始图片的地址
    data = prepare_data(sess, dataset=data_dir)
  else:
    data = prepare_data(sess, dataset=data_dir)

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(config.image_size - config.label_size) / 2

  if config.is_train:
    for i in range(len(data)):
      input_=(imread(data[i])-127.5)/127.5
      label_=input_

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      #按14步长采样小patch
      for x in range(0, h-config.image_size+1, config.stride):
        for y in range(0, w-config.image_size+1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size]
          #注意这里的padding，前向传播时由于卷积是没有padding的，所以实际上预测的是测试patch的中间部分
          sub_label = label_[int(x+padding):int(x+padding+config.label_size), int(y+padding):int(y+padding+config.label_size)]
          # Make channel value
          if data_dir == "Train":
            sub_input = cv2.resize(sub_input, (config.image_size/4,config.image_size/4), interpolation=cv2.INTER_CUBIC)
            sub_input = sub_input.reshape([config.image_size/4, config.image_size/4, 1])
            sub_label = cv2.resize(sub_label, (config.label_size/4,config.label_size/4), interpolation=cv2.INTER_CUBIC)
            sub_label = sub_label.reshape([config.label_size/4, config.label_size/4, 1])
            print('error')
          else:
            sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
            sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
          
          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)
  #print(arrdata.shape)
  savepath = make_data(sess, arrdata, arrlabel, data_dir)

  print('end input_setup')
  return savepath




def imsave(image, path):
  return scipy.misc.imsave(path, image)






def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return (img*127.5+127.5)
  
# def gradient(input):
#     #filter_x=tf.reshape(tf.constant([[-1.,0.,1.],[-1.,0.,1.],[-1.,0.,1.]]),[3,3,1,1])
#     #filter_y=tf.reshape(tf.constant([[-1.,-1.,-1],[0,0,0],[1,1,1]]),[3,3,1,1])
#     #d_x=tf.nn.conv2d(input,filter_x,strides=[1,1,1,1], padding='SAME')
#     #d_y=tf.nn.conv2d(input,filter_y,strides=[1,1,1,1], padding='SAME')
#     #d=tf.sqrt(tf.square(d_x)+tf.square(d_y))
#     filter=tf.reshape(tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]),[3,3,1,1])
#     d=tf.nn.conv2d(input,filter,strides=[1,1,1,1], padding='SAME')
#     return d

def gradient(img):
    kernel_x = tf.constant([[-3, 0, +3], [-10, 0, +10], [-3, 0, +3]])
    kernel_y = tf.constant([[-3, -10, -3], [0, 0, 0], [+3, +10, +3]])
    kernel_x = tf.expand_dims(kernel_x, axis=-1)
    kernel_x = tf.expand_dims(kernel_x, axis=-1)
    kernel_y = tf.expand_dims(kernel_y, axis=-1)
    kernel_y = tf.expand_dims(kernel_y, axis=-1)
    g_x = tf.nn.conv2d(img, tf.cast(kernel_x, tf.float32), strides=[1, 1, 1, 1], padding='SAME')
    g_y = tf.nn.conv2d(img, tf.cast(kernel_y, tf.float32), strides=[1, 1, 1, 1], padding='SAME')
    g = g_x + g_y
    return g
    
def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite+1
        
        u_hat, v_hat,_ = power_iteration(u,iteration)
        
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        
        w_mat = w_mat/sigma
        
        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not(update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))
            
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm
    
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)
    
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm
