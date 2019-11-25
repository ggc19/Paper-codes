# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:52:53 2019

@author: ggc
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

'''-------------------------------------read data--------------------------------------'''
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
    Data_y[i,1] = 0  
for j in range(0,5481):
    Data_y[5481+2*j,0] = Data1_y[5481+2*j]
    Data_y[5481+2*j,1] = 1  # 5dB
    Data_y[5481+2*j+1,0] = Data1_y[5481+2*j+1]
    Data_y[5481+2*j+1,1] = 2  # 10dB   
for j in range(0,5481):
    Data_y[5481*3+3*j,0] = Data1_y[5481*3+3*j]
    Data_y[5481*3+3*j,1] = 3  # -10dB
    Data_y[5481*3+3*j+1,0] = Data1_y[5481*3+3*j+1]
    Data_y[5481*3+3*j+1,1] = 4  # -5dB      
    Data_y[5481*3+3*j+2,0] = Data1_y[5481*3+3*j+2]
    Data_y[5481*3+3*j+2,1] = 5  # 0dB  

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
    
'''-----------------------------------develop the FCNN model--------------------------------------'''
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
print('FCNN test loss:', scores[0])
print('FCNN test accuracy:', scores[1])
from keras.utils import plot_model
plot_model(model, to_file='FCNN model.png',show_shapes=True)

y_prob = model.predict(X_test_freq)
y_pred = np.zeros((len(y_prob),1))
for i in range(len(y_prob)):
    if y_prob[i] <= 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1        
freq_con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('FCNN accuracy：',(freq_con_matrix[0,0]+freq_con_matrix[1,1])/len(y_pred))
freq_fpr, freq_tpr, freq_thresholds = roc_curve(y_test[:,0], y_prob)
freq_roc_auc = auc(freq_fpr, freq_tpr)
plt.figure()
plt.plot(freq_fpr, freq_tpr,label='FCNN (AUC = %0.2f)' % (10, freq_roc_auc))
plt.title('Roc Curve')
plt.legend(loc='best')
plt.show()

'''-----------------------------------calculate results--------------------------------------'''
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
    print('id：%d, accuracy：%f, specificity： %f , sensitivity： %f'%(db, accuracy,specificity, sensitivity))
