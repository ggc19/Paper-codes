# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:52:53 2019

@author: 94296
"""

import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1337)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten,Input,Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as scio
from scipy.fftpack import fft

'''-------------------------------------读取数据--------------------------------------'''
path1 = 'Data.mat'
path2 = 'Data_noise1.mat'
path3 = 'Data_noise2.mat'
data = scio.loadmat(path1)
data_noise1 = scio.loadmat(path2)
data_noise2 = scio.loadmat(path3)

data = data['Data']
data_noise1 = data_noise1['Data_noise1']
data_noise2 = data_noise2['Data_noise2']
Data_total = np.concatenate((data, data_noise1, data_noise2), axis = 0)
Data_x = Data_total[:,1:4801]
Data1_y = Data_total[:,0]

Data_y = np.zeros((5481*6,2))
for i in range(0,5481):
    Data_y[i,0] = Data1_y[i]
    Data_y[i,1] = 0  # 无噪声
for j in range(0,5481):
    Data_y[5481+2*j,0] = Data1_y[5481+2*j]
    Data_y[5481+2*j,1] = 1  # 5噪声
    Data_y[5481+2*j+1,0] = Data1_y[5481+2*j+1]
    Data_y[5481+2*j+1,1] = 2  # 10噪声   
for j in range(0,5481):
    Data_y[5481*3+3*j,0] = Data1_y[5481*3+3*j]
    Data_y[5481*3+3*j,1] = 3  # -10噪声
    Data_y[5481*3+3*j+1,0] = Data1_y[5481*3+3*j+1]
    Data_y[5481*3+3*j+1,1] = 4  # -5噪声      
    Data_y[5481*3+3*j+2,0] = Data1_y[5481*3+3*j+2]
    Data_y[5481*3+3*j+2,1] = 5  # 0噪声  

Data_x_fft = np.zeros((Data_x.shape[0],1024))
for i in range(Data_x.shape[0]):
    temp = abs(fft(Data_x[i,0:2048]))
    Data_x_fft[i,:] = temp[0:1024]   

indices = np.arange(Data_x.shape[0])
np.random.shuffle(indices)
Data_x = Data_x[indices,:]
Data_x_fft = Data_x_fft[indices,:]
Data_y = Data_y[indices,:]
Data_x = (Data_x-np.average(Data_x))/np.std(Data_x)
Data_x_fft = (Data_x_fft-np.average(Data_x_fft))/np.std(Data_x_fft)

train_samples = int(Data_x.shape[0]*0.8)
test_samples = int(Data_x.shape[0]*0.2)
X_train_freq = np.zeros((train_samples,Data_x_fft.shape[1],1))
y_train = np.zeros((train_samples,1))
X_test_freq = np.zeros((test_samples,Data_x_fft.shape[1],1))
y_test = np.zeros((test_samples,2))
for i in range(train_samples):
    X_train_freq[i,:,0] = Data_x_fft[i,:]
    y_train[i,0] = Data_y[i,0]
for j in range(test_samples):
    i=j+train_samples
    X_test_freq[j,:,0] = Data_x_fft[i,:]
    y_test[j,0] = Data_y[i,0]
    y_test[j,1] = Data_y[i,1]
    
'''-----------------------------------CNN识别模型，模型输入为频域信息--------------------------------------'''
BATCH_SIZE = 64
Output_size = 1
LR = 0.001
input1 = Input(shape=(X_train_freq.shape[1],1))
con_model1 = Conv1D(32, kernel_size=32, strides=8, padding='same', activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(input1)
con_model1 = MaxPooling1D(pool_size=32, strides=4)(con_model1)
con_model1 =Flatten()(con_model1)
dense = Dense(256, activation='relu')(con_model1)
dense = Dense(128, activation='relu')(dense)
dense = Dropout(0.25,seed=1)(dense)
output = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=input1, outputs = output) 
adam = Adam(LR)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy']) 
early_stopping = EarlyStopping(monitor='val_acc', patience=5)
history=model.fit(X_train_freq, y_train, batch_size=BATCH_SIZE,epochs=100, shuffle=True,
            callbacks=[early_stopping],  validation_split=0.2)
scores = model.evaluate(X_test_freq, y_test[:,0], verbose=1)
print('频域信息Test loss:', scores[0])
print('频域信息Test accuracy:', scores[1])
from keras.utils import plot_model
plot_model(model, to_file='频域model1.png',show_shapes=True)

y_prob = model.predict(X_test_freq)
y_pred = np.zeros((len(y_prob),1))
for i in range(len(y_prob)):
    if y_prob[i] <= 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1        
freq_con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('频域CNN识别准确率为：',(freq_con_matrix[0,0]+freq_con_matrix[1,1])/len(y_pred))
freq_fpr, freq_tpr, freq_thresholds = roc_curve(y_test[:,0], y_prob)
freq_roc_auc = auc(freq_fpr, freq_tpr)
plt.figure()
plt.plot(freq_fpr, freq_tpr,label='Freq-CNN ROC fold %d (AUC = %0.2f)' % (10, freq_roc_auc))
plt.title('Roc Curve')
plt.legend(loc='best')
plt.show()

'''-----------------------------------计算每一类的效果--------------------------------------'''
for db in range(6):
    k1=0;k2=0; 
    new_pred=[]
    new_test=[]
    for i in range(6577):
        if y_test[i,1]==db:
            k1=k1+1
            new_pred.append(y_pred[i])
            new_test.append(y_test[i,0])
            if y_test[i,0]==y_pred[i]:
                k2=k2+1
    accuracy= k2/k1
    con_matrix_temp = confusion_matrix(new_test, new_pred)   
    specificity = (con_matrix_temp[0,1])/(con_matrix_temp[0,0]+con_matrix_temp[0,1])
    sensitivity = (con_matrix_temp[1,1])/(con_matrix_temp[1,0]+con_matrix_temp[1,1])
    print('编号为：%d, accuracy为：%f, specificity为： %f , sensitivity为： %f'%(db, accuracy,specificity, sensitivity))


#'''-------------------------------------测试部分--------------------------------------'''
#testing_path1 = 'testing_data_noise.mat'
#testing_data = scio.loadmat(testing_path1)
#testing_data = testing_data['testing_data_noise']
#testing_data_x = testing_data[:,1:4801]
##testing_data_x = (testing_data_x-np.average(testing_data_x))/np.std(testing_data_x)
#X_test_time = np.zeros((720,4800,1))
#y_test = np.zeros((720,1))
#
#Data_x_fft = np.zeros((720,512))
#for i in range(720):
#    temp = abs(fft(testing_data_x[i,0:1024]))
#    Data_x_fft[i,:] = temp[0:512]  
#Data_x_fft = (Data_x_fft-np.average(Data_x_fft))/np.std(Data_x_fft)
#X_test_freq = np.zeros((720,Data_x_fft.shape[1],1))
#
#for j in range(720):
#    X_test_time[j,:,0] = testing_data_x[j,:]
#    X_test_freq[j,:,0] = Data_x_fft[j,:]
#    y_test[j,0] = testing_data[j,0]
#
#y_prob = model.predict(X_test_freq)
#y_pred = np.zeros((len(y_prob),1))
#for i in range(len(y_prob)):
#    if y_prob[i] <= 0.5:
#        y_pred[i] = 0
#    else:
#        y_pred[i] = 1        
#
#freq_con_matrix = confusion_matrix(y_test, y_pred)
#print('频域CNN识别准确率为：',(freq_con_matrix[0,0]+freq_con_matrix[1,1])/len(y_pred))
#
#def GGC1(a,b):
#    k=0
#    for i in range(a,b):
#        if y_pred[i]!= y_test[i]:
#            k=k+1
#    r = 1-(k/120)
#    con_matrix_temp = confusion_matrix(y_test[a:b,0], y_pred[a:b,0])   
#    fpr = (con_matrix_temp[0,1])/(con_matrix_temp[0,0]+con_matrix_temp[0,1])
#    tpr = (con_matrix_temp[1,1])/(con_matrix_temp[1,0]+con_matrix_temp[1,1])     
#    return r, fpr, tpr    
#
#[r1, fpr1, tpr1] = GGC1(0,120)
#print('-10dB识别准确率为：%f, fpr为: %f, tpr为: %f '%(r1, fpr1, tpr1))
#[r2, fpr2, tpr2] = GGC1(120,240)
#print('-5dB识别准确率为：%f, fpr为: %f, tpr为: %f '%(r2, fpr2, tpr2))
#[r3, fpr3, tpr3] = GGC1(240,360)
#print('0dB识别准确率为：%f, fpr为: %f, tpr为: %f '%(r3, fpr3, tpr3))
#[r4, fpr4, tpr4] = GGC1(360,480)
#print('5dB识别准确率为：%f, fpr为: %f, tpr为: %f '%(r4, fpr4, tpr4))
#[r5, fpr5, tpr5] = GGC1(480,600)
#print('10dB识别准确率为：%f, fpr为: %f, tpr为: %f '%(r5, fpr5, tpr5))
#[r6, fpr6, tpr6] = GGC1(600,720)
#print('原始数据识别准确率为：%f, fpr为: %f, tpr为: %f '%(r6, fpr6, tpr6))
#
#
#Iron_y_pred = np.zeros((240,1))
#Iron_y_test = np.zeros((240,1))
#for i in range(6):
#    Iron_y_pred[0+40*i:20+40*i,0]=y_pred[0+120*i:20+120*i,0]
#    Iron_y_pred[20+40*i:40+40*i,0]=y_pred[60+120*i:80+120*i,0]
#    Iron_y_test[0+40*i:20+40*i,0]=y_test[0+120*i:20+120*i,0]
#    Iron_y_test[20+40*i:40+40*i,0]=y_test[60+120*i:80+120*i,0]        
#Iron_con_matrix = confusion_matrix(Iron_y_pred[:,0], Iron_y_test[:,0])   
#fpr1 = (Iron_con_matrix[0,1])/(Iron_con_matrix[0,0]+Iron_con_matrix[0,1])
#tpr1 = (Iron_con_matrix[1,1])/(Iron_con_matrix[1,0]+Iron_con_matrix[1,1])  
#r1 = (Iron_con_matrix[0,0]+Iron_con_matrix[1,1])/240 
#print('铸铁管识别准确率为：%f, fpr为: %f, tpr为: %f '%(r1, fpr1, tpr1))
#
#Steel_y_pred = np.zeros((240,1))
#Steel_y_test = np.zeros((240,1))
#for i in range(6):
#    Steel_y_pred[0+40*i:20+40*i,0]=y_pred[20+120*i:40+120*i,0]
#    Steel_y_pred[20+40*i:40+40*i,0]=y_pred[80+120*i:100+120*i,0]
#    Steel_y_test[0+40*i:20+40*i,0]=y_test[20+120*i:40+120*i,0]
#    Steel_y_test[20+40*i:40+40*i,0]=y_test[80+120*i:100+120*i,0]        
#Steel_con_matrix = confusion_matrix(Steel_y_pred[:,0], Steel_y_test[:,0])   
#fpr2 = (Steel_con_matrix[0,1])/(Steel_con_matrix[0,0]+Steel_con_matrix[0,1])
#tpr2 = (Steel_con_matrix[1,1])/(Steel_con_matrix[1,0]+Steel_con_matrix[1,1])  
#r2 = (Steel_con_matrix[0,0]+Steel_con_matrix[1,1])/240 
#print('刚管识别准确率为：%f, fpr为: %f, tpr为: %f '%(r2, fpr2, tpr2))
#
#P_y_pred = np.zeros((240,1))
#P_y_test = np.zeros((240,1))
#for i in range(6):
#    P_y_pred[0+40*i:20+40*i,0]=y_pred[40+120*i:60+120*i,0]
#    P_y_pred[20+40*i:40+40*i,0]=y_pred[100+120*i:120+120*i,0]
#    P_y_test[0+40*i:20+40*i,0]=y_test[40+120*i:60+120*i,0]
#    P_y_test[20+40*i:40+40*i,0]=y_test[100+120*i:120+120*i,0]        
#P_con_matrix = confusion_matrix(P_y_pred[:,0], P_y_test[:,0])   
#fpr3 = (P_con_matrix[0,1])/(P_con_matrix[0,0]+P_con_matrix[0,1])
#tpr3 = (P_con_matrix[1,1])/(P_con_matrix[1,0]+P_con_matrix[1,1])  
#r3 = (P_con_matrix[0,0]+P_con_matrix[1,1])/240 
#print('塑料管识别准确率为：%f, fpr为: %f, tpr为: %f '%(r3, fpr3, tpr3))
#
#
#'''-------------------------------------测试部分--------------------------------------'''
#testing_path1 = 'testing_data_noise.mat'
#testing_data = scio.loadmat(testing_path1)
#testing_data = testing_data['testing_data_noise']
#testing_data_x = testing_data[600:720,1:4801]
#y_test = np.zeros((120,1))
#
#Data_x_fft = np.zeros((120,512))
#for i in range(120):
#    temp = abs(fft(testing_data_x[i,0:1024]))
#    Data_x_fft[i,:] = temp[0:512]  
#Data_x_fft = (Data_x_fft-np.average(Data_x_fft))/np.std(Data_x_fft)
#X_test_freq = np.zeros((120,Data_x_fft.shape[1],1))
#
#for j in range(120):
#    X_test_freq[j,:,0] = Data_x_fft[j,:]
#    y_test[j,0] = testing_data[j,0]
#
#y_prob = model.predict(X_test_freq)
#y_pred = np.zeros((len(y_prob),1))
#for i in range(len(y_prob)):
#    if y_prob[i] <= 0.5:
#        y_pred[i] = 0
#    else:
#        y_pred[i] = 1        
#
#con_matrix = confusion_matrix(y_test, y_pred)
#print('Aaauracy：',(con_matrix[0,0]+con_matrix[1,1])/len(y_pred))
#print('Sensitivity：',(con_matrix[1,1])/(con_matrix[1,0]+con_matrix[1,1]))
#print('Specificity：',(con_matrix[0,1])/(con_matrix[0,0]+con_matrix[0,1]))