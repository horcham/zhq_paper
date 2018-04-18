#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import RandomState
import sys
from base_function import *
from write_submission import *
from data_process import *
from test import *

'''
指数衰减学习率，L1正则化,带偏置bias,sigmoid

接受输入：
BATCH_SIZE:int,batch数目
TRAINING_STEPS:int,训练次数
LERANING_RATE_BASE:float32,基础的学习率
LEARNING_RATE_DECAY:float32,学习率的衰减率
REGULARIZER_RATE:float,正则化项在损失函数中的系数
TOLERANCE:float32,容许误差
MOVING_AVERAGE_DECAY:float32,滑动平均损失
CONV1_SIZE:卷积层1的核大小
CONV1_DEEP:卷积层1的核数目
CONV2_SIZE:卷积层2的核大小
CONV2_DEEP:卷积层2的核数目
FC_SIZE:全连接层的隐含层大小
TEST_SECONDS:测试时间间隔
'''

#接收参数输入
TEST_SECONDS = np.int(sys.argv[1])
# 读取数据，并用自助法抽样
x_tr_mat,y_tr_mat,x_va_mat,y_va_mat,x_te_mat = read_data('/media/horcham/E/kaggle/Plant Seedlings Classification/data_tr_va','data3D_rectangle_64_64_conv')



# 产生训练需要的超参数
SAMPLE_NUM = x_tr_mat.shape[0]
NUM_LABELS = len(np.unique(y_tr_mat))
NUM_CHANNELS = x_tr_mat.shape[3]
TEST_SIZE = x_va_mat.shape[0]
MODEL_PATH = "/media/horcham/E/kaggle/Plant Seedlings Classification/code/VGGNet/model"
MODEL_NAME = "model.ckpt"

#标签onehot处理

t_va_tensor = tf.one_hot(y_va_mat,NUM_LABELS,1,0)
t_va_tensor = tf.reshape(t_va_tensor,[y_va_mat.shape[0],NUM_LABELS])
t_va_tensor = tf.cast(t_va_tensor,dtype=tf.float32)



test(x_va_mat,t_va_tensor,x_te_mat,TEST_SIZE,NUM_LABELS,MODEL_PATH,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP,CONV2_SIZE,CONV2_DEEP,CONV3_SIZE,CONV3_DEEP,CONV4_SIZE,CONV4_DEEP,FC1_SIZE,FC2_SIZE,TEST_SECONDS)
