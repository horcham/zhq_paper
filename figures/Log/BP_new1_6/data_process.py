# coding:utf-8
import numpy as np
from base_function import *
import os


def arraysave(data,filename,path):
    """
    filename:str,文件名
    path:str,路径名称,如'/media/horcham/新加卷/kaggle/Plant Seedlings Classification/train'
    """
    np.save(path + '/' + filename + '.npy', data)

def split_data(tr_file,tr_output_file,te_file,data_name):
    '''
    只接受np.array3D,.npy文件，文件都存在/media/horcham/E/kaggle/Plant Seedlings Classification/data
    tr_file:str,训练数据输入，例如data3D_tr_rectangle_32_32.npy
    tr_output:str,训练数据输出
    te_file:str，测试数据输出
    data_name:str,划分后的数据储存文件名称为x_va_mat_data_name

    返回结果为通过自助法抽样的数据，有训练集、验证集、测试集，都为np.array4D形式:[samples,hieght,width,channel]
    '''
    path = '../data'

    #读取数据
    data3D_tr = np.load(path+'/'+tr_file)
#    data3D_tr = np.load(os.path.join(path,tr_file))
    data_tr_output = np.load(path+'/'+tr_output_file)
    data_tr_output = np.load(os.path.join(path,tr_output_file))
    data3D_te = np.load(os.path.join(path,te_file))
    x_te_mat = data3D_te

    x_va_mat,y_va_mat,x_tr_mat,y_tr_mat = sample_bootstrap(data3D_tr,data_tr_output)
    x_va_mat = np.reshape(x_va_mat,[x_va_mat.shape[0],x_va_mat.shape[1],x_va_mat.shape[2],1])
    x_tr_mat = np.reshape(x_tr_mat,[x_tr_mat.shape[0],x_tr_mat.shape[1],x_tr_mat.shape[2],1])
    x_te_mat = np.reshape(x_te_mat,[x_te_mat.shape[0],x_te_mat.shape[1],x_te_mat.shape[2],1])

    arraysave(x_va_mat,'x_va_mat_'+data_name, '../data_tr_va')
    arraysave(y_va_mat,'y_va_mat_'+data_name, '../data_tr_va')
    arraysave(x_tr_mat,'x_tr_mat_'+data_name, '../data_tr_va')
    arraysave(y_tr_mat,'y_tr_mat_'+data_name, '../data_tr_va')
    arraysave(x_te_mat,'x_te_mat_'+data_name, '../data_tr_va')



def read_data(path,data_name):
    x_va_mat = np.load(os.path.join(path,'x_va_mat_'+data_name+'.npy'))
    y_va_mat = np.load(os.path.join(path,'y_va_mat_'+data_name+'.npy'))
    x_tr_mat = np.load(os.path.join(path,'x_tr_mat_'+data_name+'.npy'))
    y_tr_mat = np.load(os.path.join(path,'y_tr_mat_'+data_name+'.npy'))
    x_te_mat = np.load(os.path.join(path,'x_te_mat_'+data_name+'.npy'))
    return x_va_mat,y_va_mat,x_tr_mat,y_tr_mat,x_te_mat
