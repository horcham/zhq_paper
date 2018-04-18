#coding:utf-8
import tensorflow as tf
import numpy as np

FC_SIZE = 1000
CLASS = 13

def get_weight_variable(shape,regularizer):
    con_weights = tf.get_variable("weights",shape,initializer=tf.contrib.layers.xavier_initializer())
    if regularizer != None:
        tf.add_to_collection('losses',tf.clip_by_value(regularizer(con_weights),0,2))
    return con_weights


def inference(NUM_LABELS,input_tensor,regularizer,reuse,if_train):
    
    input_shape = input_tensor.get_shape().as_list()
    nodes = input_shape[1]*input_shape[2]*input_shape[3]
    print nodes
    reshaped = tf.reshape(input_tensor,[-1,nodes])

    #全连接层
    with tf.variable_scope('fc1',reuse=reuse):
        fc_weights = get_weight_variable([nodes,FC_SIZE],regularizer)
        fc_biases = tf.get_variable('biases',[FC_SIZE],initializer=tf.constant_initializer(0.0))
        sub_out = tf.nn.relu(tf.matmul(reshaped,fc_weights)+fc_biases)
	if if_train:
            sub_out = tf.nn.dropout(sub_out,0.1)
    with tf.variable_scope('fc2',reuse=reuse):
        fc_weights = get_weight_variable([FC_SIZE,CLASS],regularizer)
        fc_biases = tf.get_variable('biases',[CLASS],initializer=tf.constant_initializer(0.0))
        out = tf.nn.relu(tf.matmul(sub_out,fc_weights)+fc_biases)
    return sub_out,out



