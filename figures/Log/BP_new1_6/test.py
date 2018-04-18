#coding:utf-8
import numpy as np
import time 
import tensorflow as tf
from inference_conv import *
import os
from data_process import *
from write_submission import *
import matplotlib.pyplot as plt 

def test(x_va,y_va_tensor,x_te,TEST_SIZE,NUM_LABELS,MODEL_PATH,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP,CONV2_SIZE,CONV2_DEEP,CONV3_SIZE,CONV3_DEEP,CONV4_SIZE,CONV4_DEEP,FC1_SIZE,FC2_SIZE,TEST_SECONDS):

    '''
    x_va:np.array2D,[samples,features]
    y_va:tf.tensor,[samples,labels]
    '''
    x_holder = tf.placeholder(tf.float32,[None,x_va.shape[1],x_va.shape[2],NUM_CHANNELS],name='x-input')
    y_holder = tf.placeholder(tf.float32,[None,NUM_LABELS],name='y-input')

    pred = inference(NUM_LABELS,x_holder,None,None,False)
    correct_prediction = tf.equal(tf.argmax(y_holder,1),tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    out = tf.argmax(pred,1)

#    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
#    saver = tf.train.Saver(variable_averages.variables_to_restore())
    saver = tf.train.Saver()
    accuracy_report = []
    with tf.Session() as sess:
        while True:
            y_va = y_va_tensor.eval()
            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                print 'path:',ckpt.model_checkpoint_path
                saver.restore(sess,ckpt.model_checkpoint_path)
                print 'restore'
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print 'split'
                accuracy_i_list = []
                for i in range(TEST_SIZE):
                    if i % 50 == 0:
                        print('accuracy_calu:%d/%d'%(i,TEST_SIZE))
                    x_va_i = np.reshape(x_va[i,:,:,:],[1,x_va.shape[1],x_va.shape[2],x_va.shape[3]])
                    y_va_i = np.reshape(y_va[i,:],[1,y_va.shape[1]])
                    accuracy_i = sess.run(accuracy,feed_dict={x_holder:x_va_i,y_holder:y_va_i})
                    accuracy_i_list.append(accuracy_i)
                accuracy_score = np.mean(np.array(accuracy_i_list)) 
                print accuracy_score
                accuracy_report.append(accuracy_score)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.plot(range(len(accuracy_report)),accuracy_report)
                plt.title('accuracy')
                plt.savefig('accuracy.png')
                print 'accuracy_score:',accuracy_score

                o_list = []
                for i in range(x_te.shape[0]):
                    if i % 50 == 0:
                        print('test_calu:%d/%d'%(i,x_te.shape[0]))
                    x_te_i = np.reshape(x_te[i,:,:,:],[1,x_te.shape[1],x_te.shape[2],x_te.shape[3]])
                    o_i = sess.run(out,feed_dict={x_holder:x_te_i})
#                    print 'o_i:',o_i
                    o_list.append(o_i)
                o_list = np.hstack(o_list)
                write_csv(o_list)
                print 'write_csv'
                print 'sleep'
                time.sleep(TEST_SECONDS)
            else:
                print('No checkpoint file found')
                return -1
                
