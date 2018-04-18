#coding:utf-8
import os
import tensorflow as tf
from inference_conv import *
from base_function import *
#from test import *
import matplotlib.pyplot as plt
import time

NUM_CHANNELS = 1
def MyCapper(grad):
    return tf.clip_by_norm(grad,1)


def train(x_tr,y_tr,x_va,y_va,x_te,BATCH_SIZE,TEST_SIZE,EPOCH,SAMPLE_NUM,NUM_LABELS,REGULARIZER_RATE,LEARNING_RATE,MODEL_SAVE_PATH,MODEL_NAME,TIMEPNG_NAME,ACCPNG_NAME,LOG_NAME,PREOUT_NAME):
    '''
    x_tr:np.array4D,[samples,height,width,channel]
    y_tr_tensor:tf.tensor,[samples,labels]
    BATCH_SIZE:int,训练批量
    TRAINING_STEPS:int,训练次数
    SAMPLE_NUM:int,训练数据个数
    NUM_LABELS:int,类别数
    REGULARIZER_RATE:np.float,正则化系数
    MOVING_AVERAGE_DECAY:np.float,滑动平均损失
    LEARNING_RATE_BASE:np.float,基本学习率
    sdasdsaxzcxzssqssLEARNING_RATE_DECAY:np.float,学习率损失率
    MODEL_SAVE_PATH:模型文件存放路径
    MODEL_NAME:模型名称
    CONV1_SIZE:int,卷积层1核的大小
    NUM_CHANNELS:int,输入数据的通道数
    CONV1_DEEP:int,sadsadsad:w
    
    卷积层1核的数目
    CONV2_SIZE:int,卷积层2核的大小
    CONV2_DEEP:int,卷积层2核的数目
    FC_SIZE:int,全连接层隐含层大小
    '''
    MODEL_PATH = MODEL_SAVE_PATH
    accuracy_report = []
    loss_report = []
    time_report = []

    g2 = tf.Graph()
    with g2.as_default():
        x_holder = tf.placeholder(tf.float32,[None,x_tr.shape[1],x_tr.shape[2],NUM_CHANNELS],name='x-input')
        y_holder = tf.placeholder(tf.float32,[None,NUM_LABELS],name='y-input')
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
        sub_pred,pred = inference(NUM_LABELS,x_holder,regularizer,False,True)
        

    #    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #    variable_averages_op = variable_averages.apply(tf.trainable_variables())

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=tf.argmax(y_holder,1))
        cross_entropy_sum = tf.reduce_sum(cross_entropy)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,int(SAMPLE_NUM/BATCH_SIZE),LEARNING_RATE_DECAY)
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        grads_and_vars = opt.compute_gradients(loss,tf.trainable_variables())
        capped_grads_and_vars = [(MyCapper(gv[0]),gv[1]) for gv in grads_and_vars]
    #    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss,global_step=global_step)
        train_op = opt.apply_gradients(capped_grads_and_vars)
        correction_prediction = tf.equal(tf.argmax(y_holder,1),tf.argmax(pred,1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))
        out = tf.argmax(pred,1)
        init = tf.initialize_all_variables()
        g2.finalize()
    #    with tf.control_dependencies([train_step,variable_averages_op]):
    #        train_op = tf.no_op(name='train')

    #    saver = tf.train.Saver()

    with tf.Session(graph=g2) as sess2:
        sess2.run(init)
        epoch = 0
        max_batch = x_tr.shape[0]/BATCH_SIZE
        while epoch <= EPOCH:
            batch = 0
            while batch*BATCH_SIZE <= x_tr.shape[0]:
                if (batch+1)*BATCH_SIZE <= x_tr.shape[0]:
                    xs = x_tr[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE,:]
                    ys = y_tr[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE,:]
                    batch += 1
                else:
                    xs = x_tr[batch*BATCH_SIZE:x_tr.shape[0],:]
                    ys = y_tr[batch*BATCH_SIZE:x_tr.shape[0],:]
                    batch += 1
                    print('will break')
                    
                
                start_time = time.time()
                _,cem_value,loss_value = sess2.run([train_op,cross_entropy_mean,loss],feed_dict={x_holder:xs,y_holder:ys})
                end_time = time.time()

    #   测试与输出频率
                if batch % 100 == 0:
                    log = 'epoch:%d/%d    batch:%d/%d    lr:%s    cem:%f   ce+l2:%f   time:%f' % (epoch,EPOCH,batch,max_batch,LEARNING_RATE,cem_value,loss_value,end_time-start_time)
                    print log
                    f = open(LOG_NAME,'a')
                    f.write(log+'\n')
                    f.close()
            epoch += 1

            accuracy_i_list = []
            for i in range(TEST_SIZE):
                if i % 500 == 0:
                    print('accuracy_calu:%d/%d'%(i,TEST_SIZE))
                x_va_i = np.reshape(x_va[i,:,:,:],[1,x_va.shape[1],x_va.shape[2],x_va.shape[3]])
                y_va_i = np.reshape(y_va[i,:],[1,y_va.shape[1]])
                accuracy_i = sess2.run(accuracy,feed_dict={x_holder:x_va_i,y_holder:y_va_i})
                accuracy_i_list.append(accuracy_i)
            accuracy_score = np.mean(np.array(accuracy_i_list))
            print accuracy_score
            f = open(LOG_NAME,'a')
            f.write(str(accuracy_score)+'\n')
            f.close()
            accuracy_report.append(accuracy_score)
            plot_report1(accuracy_report,ACCPNG_NAME)
#                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


        sub_list = []
        for i in range(x_tr.shape[0]):
            if i % 500 == 0:
                print('o_list:%d/%d'%(i,x_tr.shape[0]))
            o_i_x = np.reshape(x_tr[i,:,:,:],[1,x_tr.shape[1],x_tr.shape[2],x_tr.shape[3]])    
            o_i_y = np.reshape(y_tr[i,:],[1,y_tr.shape[1]])
            sub_i = sess2.run(sub_pred,feed_dict={x_holder:o_i_x,y_holder:o_i_y})
            sub_list.append(sub_i)
        sub_array = np.vstack(sub_list)
        np.save(PREOUT_NAME,sub_array)



