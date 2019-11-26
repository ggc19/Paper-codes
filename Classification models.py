# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:52:18 2019

@author: ggc
"""
import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import scipy.io as scio
from sklearn.neural_network import MLPClassifier
# Gridsearch function and 5-fold corss validaiton to obtain the suitable parameters. Here, we present the optimized parameters. 

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

path11 = 'Feature.mat'
path22 = 'Feature_noise1.mat'
path33 = 'Feature_noise2.mat'
feature = scio.loadmat(path11)
feature_noise1 = scio.loadmat(path22)
feature_noise2 = scio.loadmat(path33)
feature = feature['Feature']
feature_noise1 = feature_noise1['Feature_noise1']
feature_noise2 = feature_noise2['Feature_noise2']
Feature_total = np.concatenate((feature, feature_noise1, feature_noise2), axis = 0)

indices = np.arange(Feature_total.shape[0])
np.random.shuffle(indices)
Feature_total = Feature_total[indices,0:14]  
Data_y = Data_y[indices,:]
for i in range(Feature_total.shape[1]):
    temp = Feature_total[:,i]
    Feature_total[:,i] = (temp-np.average(temp))/np.std(temp)
train_samples = int(Feature_total.shape[0]*0.8)
test_samples = int(Feature_total.shape[0]*0.2)
X_train = np.zeros((train_samples,Feature_total.shape[1]))
y_train = np.zeros((train_samples,1))
X_test = np.zeros((test_samples,Feature_total.shape[1]))
y_test = np.zeros((test_samples,2))
for i in range(train_samples):
    X_train[i,:] = Feature_total[i,:]
    y_train[i,0] = Data_y[i,0]
for j in range(test_samples):
    i=j+train_samples
    X_test[j,:] = Feature_total[i,:]
    y_test[j,0] = Data_y[i,0]
    y_test[j,1] = Data_y[i,1]

'''-------------------------------------XGBoost model--------------------------------------'''
param_dist = {'objective':'binary:logistic', 'n_estimators':2000,'learning_rate':0.05,'max_depth':8}
xg_clf = XGBClassifier(**param_dist)
xg_clf.fit(X_train, y_train)
y_pred = xg_clf.predict(X_test)
y_prob = xg_clf.predict_proba(X_test)
XG_features = xg_clf.feature_importances_
XG_con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('XGBoost accuracy：',(XG_con_matrix[0,0]+XG_con_matrix[1,1])/len(y_pred))
XG_fpr, XG_tpr, XG_thresholds = roc_curve(y_test[:,0], y_prob[:,1])
XG_roc_auc = auc(XG_fpr, XG_tpr)

'''-------------------------------------RandomForest model--------------------------------------'''
RF_clf=RandomForestClassifier(n_estimators=1000, max_features='auto', min_samples_split=2,
                              min_samples_leaf=1, criterion='entropy', bootstrap=True)
RF_clf.fit(X_train, y_train)
y_pred = RF_clf.predict(X_test)
y_prob = RF_clf.predict_proba(X_test)
RF_features = RF_clf.feature_importances_
RF_con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('RandomForest accuracy：',(RF_con_matrix[0,0]+RF_con_matrix[1,1])/len(y_pred))
RF_fpr, RF_tpr, RF_thresholds = roc_curve(y_test[:,0], y_prob[:,1])
RF_roc_auc = auc(RF_fpr, RF_tpr)

'''-------------------------------------------SVM model----------------------------------------'''
SVM_clf = SVC(kernel="rbf", C=1000, gamma="scale", probability=True)
SVM_clf.fit(X_train, y_train)
y_pred = SVM_clf.predict(X_test)
y_prob = SVM_clf.predict_proba(X_test)
SVM_con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('SVM accuracy：',(SVM_con_matrix[0,0]+SVM_con_matrix[1,1])/len(y_pred))
SVM_fpr, SVM_tpr, SVM_thresholds = roc_curve(y_test[:,0], y_prob[:,1])
SVM_roc_auc = auc(SVM_fpr, SVM_tpr)

'''-----------------------------------------DecesionTree model----------------------------------'''
DT_clf = DecisionTreeClassifier(criterion='entropy',max_depth=100,min_samples_split=4,
                                min_samples_leaf=2)
DT_clf.fit(X_train, y_train)
y_pred = DT_clf.predict(X_test)
y_prob = DT_clf.predict_proba(X_test)
DT_con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('DT accuracy：',(DT_con_matrix[0,0]+DT_con_matrix[1,1])/len(y_pred))
DT_fpr, DT_tpr, DT_thresholds = roc_curve(y_test[:,0], y_prob[:,1])
DT_roc_auc = auc(DT_fpr, DT_tpr)

'''-----------------------------------------MLP model----------------------------------'''
MLP_clf = MLPClassifier(hidden_layer_sizes=(128,),solver='adam',max_iter=2000, alpha=1e-4, random_state=1)
MLP_clf.fit(X_train, y_train)
y_pred = MLP_clf.predict(X_test)
y_prob = MLP_clf.predict_proba(X_test)
MLP_con_matrix = confusion_matrix(y_test[:,0], y_pred)
print('MLP accuracy：',(MLP_con_matrix[0,0]+MLP_con_matrix[1,1])/len(y_pred))
MLP_fpr, MLP_tpr, MLP_thresholds = roc_curve(y_test[:,0], y_prob[:,1])
MLP_roc_auc = auc(MLP_fpr, MLP_tpr)

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

'''-------------------------------------ROC curve-------------------------------------'''
excel_path= 'roc curve.csv'
roc_data = pd.read_csv(excel_path)
plt.figure(dpi=300)
plt.plot(roc_data['TFCNN-fpr'][0:185],roc_data['TFCNN-tpr'][0:185],'-', label='TFCNN model  AUC=0.99')
plt.plot(roc_data['TCNN-fpr'][0:973],roc_data['TCNN-tpr'][0:973],'--',label='TCNN model AUC=0.97')
plt.plot(roc_data['FCNN-fpr'][0:806],roc_data['FCNN-tpr'][0:806],'--',label='FCNN model AUC=0.98')
plt.plot(roc_data['DT-fpr'][0:6],roc_data['DT-tpr'][0:6],'--',label='DT model  AUC=0.78')
plt.plot(roc_data['SVM-fpr'][0:1493],roc_data['SVM-tpr'][0:1493],'--',label='SVM model AUC=0.92')
plt.plot(roc_data['MLP-fpr'][0:1588],roc_data['MLP-tpr'][0:1588],'--',label='MLP model AUC=0.91')
plt.plot(roc_data['RF-fpr'][0:906],roc_data['RF-tpr'][0:906],'--',label='RF model AUC=0.94')
plt.plot(roc_data['XG-fpr'][0:1224],roc_data['XG-tpr'][0:1224],'--', label='XGBoost model AUC=0.94')
plt.title('ROC Curve', fontdict={'family' : 'Times New Roman', 'size': 6})
plt.ylabel('TPR',fontdict={'family' : 'Times New Roman', 'size': 6})
plt.xlabel('FPR',fontdict={'family' : 'Times New Roman', 'size': 6})
plt.yticks(fontproperties = 'Times New Roman', size = 6)
plt.xticks(fontproperties = 'Times New Roman', size = 6)
plt.legend(loc='best',prop={'family' : 'Times New Roman', 'size': 6})
plt.show()

