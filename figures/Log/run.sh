#图片resize成指定大小
#ipython read_data.py 64 64
#输出文件为data_tr_64_64.npy,data_tr_o_64_64.npy,data_te_64_64.npy,位于../data 
#ipython read_data.py 64 64


#创建储存模型报告文件夹
#mkdir ./Log/VGG1
mkdir ./Log/BP_new1

#VGG
#ipython main.py [BATCH_SIZE batch大小] [EPOCH 迭代次数] [LEARNING_RATE学习率] [REGULARIZER_RATE 正则化系数] [TOLERANCE 容许误差] [TR_INPUT 训练输入] [TR_OUTPUT 训练输出] [TE_INPUT 测试输入] [TIMEPNG_NAME 训练时间文件名] [ACCPNG_NAME 准确率文件名] [LOG_NAME 日志名]
#sudo python VGGNet/main.py 8 1 0.05 0.05 0.0001 data_tr_64_64.npy data_tr_o_64_64.npy data_te_64_64.npy ./Log/VGG1/VGGNet_time.png ./Log/VGG1/VGGNet_acc.png ./Log/VGG1/VGGNet.log
sudo python BP_new1/main.py 8 600 0.03 0.000001 0.0001 data_tr_64_64.npy data_tr_o_64_64.npy data_te_64_64.npy ./Log/BP_new1/BP_new1_time.png ./Log/BP_new1/BP_new1_acc.png ./Log/BP_new1/BP_new1.log ./Log/BP_new1/BP_new1_predout.npy



mkdir ./Log/BP_new2
sudo python BP_new2/main.py 8 600 0.03 0.000001 0.0001 data_tr_96_96.npy data_tr_o_96_96.npy data_te_96_96.npy ./Log/BP_new2/BP_new2_time.png ./Log/BP_new2/BP_new2_acc.png ./Log/BP_new2/BP_new2.log ./Log/BP_new2/BP_new2_predout.npy

#ipython VGGNet/main.py 8 2 0.01 0.05 0.0001 data_tr_64_64.npy data_tr_o_64_64.npy data_te_64_64.npy ./Log/VGG2/VGGNet_time.png ./Log/VGG2/VGGNet_acc.png ./Log/VGG2/VGGNet.log


mkdir ./Log/BP_new3
sudo python BP_new3/main.py 8 600 0.03 0.000001 0.0001 data_tr_128_128.npy data_tr_o_128_128.npy data_te_128_128.npy ./Log/BP_new3/BP_new3_time.png ./Log/BP_new3/BP_new3_acc.png ./Log/BP_new3/BP_new3.log ./Log/BP_new3/BP_new3_predout.npy

mkdir ./Log/BP_new4
sudo python BP_new4/main.py 8 600 0.03 0.000001 0.0001 data_tr_64_64.npy data_tr_o_64_64.npy data_te_64_64.npy ./Log/BP_new4/BP_new4_time.png ./Log/BP_new4/BP_new4_acc.png ./Log/BP_new4/BP_new4.log ./Log/BP_new4/BP_new4_predout.npy

mkdir ./Log/BP_new5
sudo python BP_new5/main.py 8 600 0.03 0.000001 0.0001 data_tr_96_96.npy data_tr_o_96_96.npy data_te_96_96.npy ./Log/BP_new5/BP_new5_time.png ./Log/BP_new5/BP_new5_acc.png ./Log/BP_new5/BP_new5.log ./Log/BP_new5/BP_new5_predout.npy

mkdir ./Log/BP_new6
sudo python BP_new6/main.py 8 600 0.03 0.000001 0.0001 data_tr_128_128.npy data_tr_o_128_128.npy data_te_128_128.npy ./Log/BP_new6/BP_new6_time.png ./Log/BP_new6/BP_new6_acc.png ./Log/BP_new6/BP_new6.log ./Log/BP_new6/BP_new6_predout.npy

mkdir ./Log/BP_new7
sudo python BP_new7/main.py 8 600 0.03 0.1 0.0001 data_tr_96_96.npy data_tr_o_96_96.npy data_te_96_96.npy ./Log/BP_new7/BP_new7_time.png ./Log/BP_new7/BP_new7_acc.png ./Log/BP_new7/BP_new7.log ./Log/BP_new7/BP_new7_predout.npy

mkdir ./Log/BP_new8
sudo python BP_new8/main.py 8 600 0.03 0.01 0.0001 data_tr_96_96.npy data_tr_o_96_96.npy data_te_96_96.npy ./Log/BP_new8/BP_new8_time.png ./Log/BP_new8/BP_new8_acc.png ./Log/BP_new8/BP_new8.log ./Log/BP_new8/BP_new8_predout.npy

mkdir ./Log/BP_new9
sudo python BP_new9/main.py 8 600 0.03 0.001 0.0001 data_tr_96_96.npy data_tr_o_96_96.npy data_te_96_96.npy ./Log/BP_new9/BP_new9_time.png ./Log/BP_new9/BP_new9_acc.png ./Log/BP_new9/BP_new9.log ./Log/BP_new9/BP_new9_predout.npy

mkdir ./Log/BP_new10
sudo python BP_new10/main.py 8 600 0.03 0.0001 0.0001 data_tr_96_96.npy data_tr_o_96_96.npy data_te_96_96.npy ./Log/BP_new10/BP_new10_time.png ./Log/BP_new10/BP_new10_acc.png ./Log/BP_new10/BP_new10.log ./Log/BP_new10/BP_new10_predout.npy
