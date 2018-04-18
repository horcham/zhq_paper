# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

reload(sys)
sys.setdefaultencoding('gbk')

def sample_bootstrap(x,y,rate=0.33):
    '''
    自助法采样，分割验证集和训练集  
    x:[samples,height,width],np.array3D
    y:[samples,labels],np.array2D
    rate:float
    '''
    m,n,c = x.shape
    va_index = np.random.randint(0,m,np.int(m*rate)).tolist()
    tr_index = [ i for i in xrange(m) if i not in va_index ]
    va_x = x[va_index,:,:]
    va_y = y[va_index,:]
    tr_x = x[tr_index,:,:]
    tr_y = y[tr_index,:]
    
    return va_x,va_y,tr_x,tr_y

def minibatch(x,y,batch_size):
    '''
    样本批量
    x:[samples,height,width],np.array3D
    y:[samples,labels],np.array2D
    batch_size:np.int

    返回
    mini_x:[samples,height,width],np.array3D
    mini_y:[samples,labels],np.array2D
    '''
    m,n,c,_ = x.shape
    index = np.random.choice(m,batch_size,replace=False).tolist()
    mini_x = x[index,:,:,:]
    mini_y = y[index,:]
    return mini_x,mini_y

def plot_report1(report,name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(report)),report)
    plt.savefig(name)
    plt.close()

def plot_report2(report1,report2,name='MSE_CE'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(report1)), report1)
    ax1.set_ylabel('MSE')
    ax1.set_title("iterator")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(range(len(report2)), report2, 'r')
    ax2.set_ylim([0, 20])
    ax2.set_ylabel('cross entropy')
    ax2.set_xlabel('iterator')
    plt.savefig('report.png')
    plt.show()

