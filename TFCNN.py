# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:56:00 2019

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
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,concatenate,Input,Dropout,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as scio

'''------------------------------------read data--------------------------------------'''
def load_data(path1_x, path1_y, data1_x_title, data1_y_title,
              path1_noise1_x, path1_noise1_y, data1_noise1_x_title, data1_noise1_y_title,
              path1_noise2_x, path1_noise2_y, data1_noise2_x_title, data1_noise2_y_title
              ):    
    path1_x = path1_x
    path1_y = path1_y
    data1_x = scio.loadmat(path1_x)
    data1_y = scio.loadmat(path1_y)
    Data1_x = data1_x[data1_x_title]
    Data1_y = data1_y[data1_y_title]
    
    path1_noise1_x = path1_noise1_x
    path1_noise1_y = path1_noise1_y
    data1_noise1_x = scio.loadmat(path1_noise1_x)
    data1_noise1_y = scio.loadmat(path1_noise1_y)
    Data1_noise1_x = data1_noise1_x[data1_noise1_x_title]
    Data1_noise1_y = data1_noise1_y[data1_noise1_y_title]
    
    path1_noise2_x = path1_noise2_x
    path1_noise2_y = path1_noise2_y
    data1_noise2_x = scio.loadmat(path1_noise2_x)
    data1_noise2_y = scio.loadmat(path1_noise2_y)
    Data1_noise2_x = data1_noise2_x[data1_noise2_x_title]
    Data1_noise2_y = data1_noise2_y[data1_noise2_y_title]
    
    Data_total_x = np.concatenate((Data1_x, Data1_noise1_x, Data1_noise2_x), axis = 0)
    Data_total_y = np.concatenate((Data1_y, Data1_noise1_y, Data1_noise2_y), axis = 0)
    return  Data_total_x, Data_total_y

Data1_x, Data1_y = load_data('Data1.mat', 'Data1_y.mat', 'Data1', 'Data1_y',
                             'Data1_noise1.mat', 'Data1_noise1_y.mat', 'Data1_noise1', 'Data1_noise1_y',
                             'Data1_noise2.mat', 'Data1_noise2_y.mat', 'Data1_noise2', 'Data1_noise2_y')

Data2_x, Data2_y = load_data('Data2.mat', 'Data2_y.mat', 'Data2', 'Data2_y',
                             'Data2_noise1.mat', 'Data2_noise1_y.mat', 'Data2_noise1', 'Data2_noise1_y',
                             'Data2_noise2.mat', 'Data2_noise2_y.mat', 'Data2_noise2', 'Data2_noise2_y')                              

Data3_x, Data3_y = load_data('Data3.mat', 'Data3_y.mat', 'Data3', 'Data3_y',
                             'Data3_noise1.mat', 'Data3_noise1_y.mat', 'Data3_noise1', 'Data3_noise1_y',
                             'Data3_noise2.mat', 'Data3_noise2_y.mat', 'Data3_noise2', 'Data3_noise2_y') 

Data_y = np.zeros((5481*6,2))
'''------------------------------------add WGN--------------------------------------'''
for i in range(0,5481):
    Data_y[i,0] = Data1_y[i,0]
    Data_y[i,1] = 0  
for j in range(0,5481):
    Data_y[5481+2*j,0] = Data1_y[5481+2*j,0]
    Data_y[5481+2*j,1] = 1  # 5dB
    Data_y[5481+2*j+1,0] = Data1_y[5481+2*j+1,0]
    Data_y[5481+2*j+1,1] = 2  # 10dB   
for j in range(0,5481):
    Data_y[5481*3+3*j,0] = Data1_y[5481*3+3*j,0]
    Data_y[5481*3+3*j,1] = 3  # -10dB
    Data_y[5481*3+3*j+1,0] = Data1_y[5481*3+3*j+1,0]
    Data_y[5481*3+3*j+1,1] = 4  # -5dB      
    Data_y[5481*3+3*j+2,0] = Data1_y[5481*3+3*j+2,0]
    Data_y[5481*3+3*j+2,1] = 5  # 0dB 

indices = np.arange(Data1_x.shape[0])
np.random.shuffle(indices)
Data1_x=Data1_x[indices,:,:]
Data2_x=Data2_x[indices,:,:]
Data3_x=Data3_x[indices,:,:]
Data_y = Data_y[indices,:]
train_samples = int(Data1_x.shape[0]*0.8)
test_samples = int(Data1_x.shape[0]*0.2)
X_train_1 = np.zeros((train_samples,Data1_x.shape[1],Data1_x.shape[2],1))
X_train_2 = np.zeros((train_samples,Data2_x.shape[1],Data2_x.shape[2],1))
X_train_3 = np.zeros((train_samples,Data3_x.shape[1],Data3_x.shape[2],1))
y_train = np.zeros((train_samples,1))
X_test_1 = np.zeros((test_samples,Data1_x.shape[1],Data1_x.shape[2],1))
X_test_2 = np.zeros((test_samples,Data2_x.shape[1],Data2_x.shape[2],1))
X_test_3 = np.zeros((test_samples,Data3_x.shape[1],Data3_x.shape[2],1))
y_test = np.zeros((test_samples,2))

for i in range(train_samples):
    Data1_x[i,:,:] = (Data1_x[i,:,:]-np.average(Data1_x[i,:,:]))/np.std(Data1_x[i,:,:])
    Data2_x[i,:,:] = (Data2_x[i,:,:]-np.average(Data2_x[i,:,:]))/np.std(Data2_x[i,:,:])
    Data3_x[i,:,:] = (Data3_x[i,:,:]-np.average(Data3_x[i,:,:]))/np.std(Data3_x[i,:,:])
    X_train_1[i,:,:,0] = Data1_x[i,:,:]
    X_train_2[i,:,:,0] = Data2_x[i,:,:]
    X_train_3[i,:,:,0] = Data3_x[i,:,:] 
    y_train[i,0] = Data_y[i,0]
    
for j in range(test_samples):
    i=j+train_samples
    Data1_x[i,:,:] = (Data1_x[i,:,:]-np.average(Data1_x[i,:,:]))/np.std(Data1_x[i,:,:])
    Data2_x[i,:,:] = (Data2_x[i,:,:]-np.average(Data2_x[i,:,:]))/np.std(Data2_x[i,:,:])
    Data3_x[i,:,:] = (Data3_x[i,:,:]-np.average(Data3_x[i,:,:]))/np.std(Data3_x[i,:,:])
    X_test_1[j,:,:,0] = Data1_x[i,:,:]
    X_test_2[j,:,:,0] = Data2_x[i,:,:]
    X_test_3[j,:,:,0] = Data3_x[i,:,:] 
    y_test[j,0] = Data_y[i,0]
    y_test[j,1] = Data_y[i,1]    
        
'''-----------------------------------develop the TFCNN model--------------------------------------'''
BATCH_SIZE = 64
Output_size = 1
LR = 0.001
input1 = Input(shape=(Data1_x.shape[1],Data1_x.shape[2],1))
con_model1 = Conv2D(32, kernel_size=(4,4), strides=(4,4), padding='same', activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(input1)
con_model1 = Conv2D(32, kernel_size=(2,2), strides=(2,2), padding='same', activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(con_model1)
con_model1 = MaxPooling2D(pool_size=(2,2))(con_model1)
con_model1 = BatchNormalization()(con_model1)
con_model1 =Flatten()(con_model1)

input2 = Input(shape=(Data2_x.shape[1],Data2_x.shape[2],1))
con_model2 = Conv2D(32, kernel_size=(4,4), strides=(4,4), padding='same', activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(input2)
con_model2 = Conv2D(32, kernel_size=(2,2), strides=(2,2), padding='same', activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(con_model2)
con_model2 = MaxPooling2D(pool_size=(2,2))(con_model2)
con_model2 = BatchNormalization()(con_model2)
con_model2 =Flatten()(con_model2)

input3 = Input(shape=(Data3_x.shape[1],Data3_x.shape[2],1))
con_model3 = Conv2D(32, kernel_size=(4,4), strides=(4,4), padding='same', activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(input3)
con_model3 = Conv2D(32, kernel_size=(2,2), strides=(2,2), padding='same', activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(con_model3)
con_model3 = MaxPooling2D(pool_size=(2,2))(con_model3)
con_model3 = BatchNormalization()(con_model3)
con_model3 =Flatten()(con_model3)

merge1 = concatenate([con_model1, con_model2, con_model3],axis = -1)
dense = Dense(512, activation='relu')(merge1)
dense = Dense(256, activation='relu')(dense)
dense = Dropout(0.25,seed=1)(dense)
output = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=[input1, input2, input3], outputs = output) 
adam = Adam(LR)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy']) 
early_stopping = EarlyStopping(monitor='val_acc', patience=10)
history=model.fit([X_train_1,X_train_2,X_train_3], y_train, batch_size=BATCH_SIZE,epochs=100, shuffle=True,
            callbacks=[early_stopping],  validation_split=0.2)

scores = model.evaluate([X_test_1,X_test_2,X_test_3], y_test[:,0], verbose=1)
print('TFCNN test loss:', scores[0])
print('TFCNN test accuracy:', scores[1])
y_prob = model.predict([X_test_1, X_test_2, X_test_3])
y_pred = np.zeros((len(y_prob),1))
for i in range(len(y_prob)):
    if y_prob[i] <= 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1        

con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('TFCNN accuracy：',(con_matrix[0,0]+con_matrix[1,1])/len(y_pred))
fpr, tpr, thresholds = roc_curve(y_test[:,0], y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr,label='TFCNN (AUC = %0.2f)' % (roc_auc))
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

from keras.utils import plot_model
plot_model(model, to_file='TFCNN model.png',show_shapes=True)

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
    print('id：%d, accuracy为：%f, specificity： %f , sensitivity： %f'%(db, accuracy,specificity, sensitivity))
