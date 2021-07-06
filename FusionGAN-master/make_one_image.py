# -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2

#reader = tf.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    #flatten=True 以灰度图的形式读取
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)


def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.png"))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='VALID') + bias,
                                                   decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5= tf.nn.conv2d(conv4, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5=tf.nn.tanh(conv5)
    return conv5





def input_setup(index, data_MRT1, data_MRT2):
    padding=4
    sub_MRT1_sequence = []
    sub_MRT2_sequence = []
    input_MRT1=(imread(data_MRT1[index])-127.5)/127.5
    input_MRT1=np.lib.pad(input_MRT1,((padding,padding),(padding,padding)),'edge')
    w,h=input_MRT1.shape
    input_MRT1=input_MRT1.reshape([w,h,1])
    input_MRT2=(imread(data_MRT2[index])-127.5)/127.5
    input_MRT2=np.lib.pad(input_MRT2,((padding,padding),(padding,padding)),'edge')
    w,h=input_MRT2.shape
    input_MRT2=input_MRT2.reshape([w,h,1])
    sub_MRT1_sequence.append(input_MRT1)
    sub_MRT2_sequence.append(input_MRT2)
    train_data_MRT1= np.asarray(sub_MRT1_sequence)
    train_data_MRT2= np.asarray(sub_MRT2_sequence)
    return train_data_MRT1,train_data_MRT2



if __name__ == '__main__':
    num_epoch=38
    while(num_epoch==38):

        reader = tf.train.NewCheckpointReader('./checkpoint_05/CGAN_248/CGAN.model-'+ str(num_epoch))

        with tf.name_scope('MRT2_input'):
            images_MRT2 = tf.placeholder(tf.float32, [1,None,None,None], name='images_MRT2')

        with tf.name_scope('MRT1_input'):
            images_MRT1 = tf.placeholder(tf.float32, [1,None,None,None], name='images_MRT1')

        with tf.name_scope('input'):
            input_image=tf.concat([images_MRT1, images_MRT2], axis=-1)

        with tf.name_scope('fusion'):
            fusion_image=fusion_model(input_image)

        with tf.Session() as sess:
            init_op=tf.global_variables_initializer()
            sess.run(init_op)
            data_MRT1=prepare_data('dataset_images/Test_MRT1')
            data_MRT2=prepare_data('dataset_images/Test_MRT2')
            for i in range(len(data_MRT1)):
                start=time.time()
                train_data_MRT1,train_data_MRT2=input_setup(i, data_MRT1, data_MRT2)
                result =sess.run(fusion_image,feed_dict={images_MRT1: train_data_MRT1,images_MRT2: train_data_MRT2})
                result=result*127.5+127.5
                result = result.squeeze()
                image_path = os.path.join(os.getcwd(), 'result_05')
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                image_path = os.path.join(image_path, str(3*(i+14))+".png")
                end=time.time()
                imsave(result, image_path)
                print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
        tf.reset_default_graph()
        num_epoch=num_epoch+1

