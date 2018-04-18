#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import RandomState
import sys
from base_function import *
from write_submission import *
from data_process import *
from train import *
#from test import *

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
FC1_SIZE:全连接层1的隐含层大小
FC2_SIZE:全连接层2的隐含层大小
'''

#接收参数输入
BATCH_SIZE = np.int(sys.argv[1]) 
EPOCH = np.int(sys.argv[2])
LEARNING_RATE = np.float32(sys.argv[3])
#LEARNING_RATE_BASE = np.float32(sys.argv[3])
#LEARNING_RATE_DECAY = np.float32(sys.argv[4])
REGULARIZER_RATE = np.float32(sys.argv[4])
TOLERANCE = np.float(sys.argv[5])
#MOVING_AVERAGE_DECAY = np.float(sys.argv[7])
TR_INPUT = sys.argv[6]
TR_OUTPUT = sys.argv[7]
TE_INPUT  = sys.argv[8]
TIMEPNG_NAME = sys.argv[9]
ACCPNG_NAME = sys.argv[10]
LOG_NAME = sys.argv[11]
PREOUT_NAME = sys.argv[12]

TEMP_data = TR_INPUT + '_temp'

f = open(LOG_NAME,'a')
print "FC_SIZE:",FC_SIZE
f.write("FC_SIZE:"+str(FC_SIZE))

# 读取数据，并用自助法抽样
#split_data('data3D_tr_rectangle_64_64.npy' ,'data_tr_output_rectangle_64_64.npy','data3D_te_rectangle_64_64.npy','data3D_rectangle_64_64_conv')              
split_data(TR_INPUT,TR_OUTPUT,TE_INPUT,TEMP_data)
#x_tr_mat,y_tr_mat,x_va_mat,y_va_mat,x_te_mat = read_data('../data_tr_va','data3D_rectangle_64_64_conv')
x_va_mat,y_va_mat,x_tr_mat,y_tr_mat,x_te_mat = read_data('../data_tr_va',TEMP_data)

print 'x_tr.shape:',x_tr_mat.shape
print 'y_tr.shape:',y_tr_mat.shape
print 'x_va.shape:',x_va_mat.shape
print 'y_va.shape:',y_va_mat.shape
print 'x_te.shape:',x_te_mat.shape

# 产生训练需要的超参数
SAMPLE_NUM = x_tr_mat.shape[0]
NUM_LABELS = len(np.unique(y_tr_mat))
NUM_CHANNELS = x_tr_mat.shape[3]
MODEL_SAVE_PATH = "./code/BP_new/model"
MODEL_NAME = "model.ckpt"
TEST_SIZE  = x_va_mat.shape[0]


#标签onehot处理
g1 = tf.Graph()
with g1.as_default():
    t_tr_tensor = tf.one_hot(y_tr_mat,NUM_LABELS,1,0)
    t_tr_tensor = tf.reshape(t_tr_tensor,[y_tr_mat.shape[0],NUM_LABELS])
    t_tr_tensor = tf.cast(t_tr_tensor,dtype=tf.float32)

    t_va_tensor = tf.one_hot(y_va_mat,NUM_LABELS,1,0)
    t_va_tensor = tf.reshape(t_va_tensor,[y_va_mat.shape[0],NUM_LABELS])
    t_va_tensor = tf.cast(t_va_tensor,dtype=tf.float32)
    g1.finalize()
with tf.Session(graph=g1) as sess1:
    t_tr,t_va = sess1.run([t_tr_tensor,t_va_tensor])

#train_test(x_tr_mat,t_tr_tensor,x_va_mat,t_va_tensor,x_te_mat,BATCH_SIZE,TEST_SIZE,TRAINING_STEPS,SAMPLE_NUM,NUM_LABELS,REGULARIZER_RATE,LEARNING_RATE,MODEL_SAVE_PATH,MODEL_NAME,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP,CONV2_SIZE,CONV2_DEEP,CONV3_SIZE,CONV3_DEEP,FC1_SIZE,FC2_SIZE)
train(x_tr_mat,t_tr,x_va_mat,t_va,x_te_mat,BATCH_SIZE,TEST_SIZE,EPOCH,SAMPLE_NUM,NUM_LABELS,REGULARIZER_RATE,LEARNING_RATE,MODEL_SAVE_PATH,MODEL_NAME,TIMEPNG_NAME,ACCPNG_NAME,LOG_NAME,PREOUT_NAME)

    

