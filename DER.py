# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:17:00 2019

@author: Sider007
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:53:35 2019

@author: Sider007
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.externals import joblib
from sklearn.metrics import recall_score
from numpy import linalg as LA

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def readdata(dataset,dataname):
    
    if(dataset == 1):
        df1 = pd.read_csv('./firstdata-train.csv',header = 0)
        data1 = df1.values

        df2 = pd.read_csv('./firstdata-test.csv',header = 0)
        data2 = df2.values
    
        folder = os.getcwd()+ '/Firstdata'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    if(dataset == 2):
        df1 = pd.read_csv('./allstate-train.csv',header = 0)
        data1 = df1.values
        
        df2 = pd.read_csv('./allstate-test.csv',header = 0)
        data2 = df2.values
        
        folder = os.getcwd() + '/Allstate'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if(dataset == 3):
        df1 = pd.read_csv('./fashion-mnist_train.csv',header = 0)
        data1 = df1.values
    
        df2 = pd.read_csv('./fashion-mnist_test.csv',header = 0)
        data2 = df2.values
        
        folder = os.getcwd() + '/FashionMNIST'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if(dataset == 4):
        df1 = pd.read_csv('./kmnist-train.csv',header = 0)
        data1 = df1.values

        df2 = pd.read_csv('./kmnist-test.csv',header = 0)
        data2 = df2.values
        
        folder = os.getcwd() + '/KuzushijiMNIST'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if(dataset == 5):
        df1 = pd.read_csv('./'+ str(dataname) +'-train.csv',header = 0)
        data1 = df1.values

        df2 = pd.read_csv('./'+ str(dataname) +'-test.csv',header = 0)
        data2 = df2.values
        
        folder = os.getcwd()[:-4] + '/'+str(dataname)
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    print(data1.shape)
    print(data2.shape)
    
    return data1, data2

def readdata_check(k,dataset,dataname):
    if(dataset == 1):
        df1 = pd.read_csv('./Firstdata/X_train_c_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_c = df1.values
        
        df1 = pd.read_csv('./Firstdata/X_train_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/y_train_c_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_c = df1.values
        
        df1 = pd.read_csv('./Firstdata/y_train_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/X_test_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        X_test = df1.values
       
        df2 = pd.read_csv('./Firstdata/y_test_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        y_test = df2.values
        
        df1 = pd.read_csv('./Firstdata/KNN_predict_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knnp_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/KNN_predict_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knnp = df1.values
        
        df1 = pd.read_csv('./Firstdata/SVC_predict_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svcp_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/SVC_predict_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svcp = df1.values
        
        df1 = pd.read_csv('./Firstdata/DEC_predict_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        decp_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/DEC_predict_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        decp = df1.values
        
        df1 = pd.read_csv('./Firstdata/XGB_predict_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/XGB_predict_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp = df1.values
        
        df1 = pd.read_csv('./Firstdata/KNN_predictpro_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/KNN_predictpro_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro = df1.values
        
        df1 = pd.read_csv('./Firstdata/SVC_predictpro_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/SVC_predictpro_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro = df1.values
        
        df1 = pd.read_csv('./Firstdata/DEC_predictpro_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        decpro_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/DEC_predictpro_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        decpro = df1.values
        
        df1 = pd.read_csv('./Firstdata/XGB_predictpro_n_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro_n = df1.values
        
        df1 = pd.read_csv('./Firstdata/XGB_predictpro_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro = df1.values
        df1 = pd.read_csv('./Firstdata/Trained test_loss_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        tra_loss = df1.values
        
        n,d = tra_loss.shape
        tra_loss_knn = tra_loss[:,0]
        tra_loss_knn = tra_loss_knn.reshape((n,1))
        tra_loss_svc = tra_loss[:,1]
        tra_loss_svc = tra_loss_svc.reshape((n,1))
        tra_loss_dec = tra_loss[:,2]
        tra_loss_dec = tra_loss_dec.reshape((n,1))
        tra_loss_xgb = tra_loss[:,3]
        tra_loss_xgb = tra_loss_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./Firstdata/Trained test_NN_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        tra_nn = df1.values
        
        n,d = tra_nn.shape
        tra_nn_knn = tra_nn[:,0]
        tra_nn_knn = tra_nn_knn.reshape((n,1))
        tra_nn_svc = tra_nn[:,1]
        tra_nn_svc = tra_nn_svc.reshape((n,1))
        tra_nn_dec = tra_nn[:,2]
        tra_nn_dec = tra_nn_dec.reshape((n,1))
        tra_nn_xgb = tra_nn[:,3]
        tra_nn_xgb = tra_nn_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./Firstdata/Trained test_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        tra_ksd = df1.values
        
        tra_ksd_knn = tra_ksd[:,0]
        tra_ksd_knn = tra_ksd_knn.reshape((n,1))
        tra_ksd_svc = tra_ksd[:,1]
        tra_ksd_svc = tra_ksd_svc.reshape((n,1))
        tra_ksd_dec = tra_ksd[:,2]
        tra_ksd_dec = tra_ksd_dec.reshape((n,1))
        tra_ksd_xgb = tra_ksd[:,3]
        tra_ksd_xgb = tra_ksd_xgb.reshape((n,1))
        
        df1 = pd.read_csv('./Firstdata/KNN_n_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knn_exac = df1.values
        df1 = pd.read_csv('./Firstdata/SVC_n_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svc_exac = df1.values
        df1 = pd.read_csv('./Firstdata/DEC_n_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        dec_exac = df1.values
        df1 = pd.read_csv('./Firstdata/XGB_n_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_exac = df1.values
        
        df1 = pd.read_csv('./Firstdata/KNN_n_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knn_ksd = df1.values
        df1 = pd.read_csv('./Firstdata/SVC_n_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svc_ksd = df1.values
        df1 = pd.read_csv('./Firstdata/DEC_n_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        dec_ksd = df1.values
        df1 = pd.read_csv('./Firstdata/XGB_n_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_ksd = df1.values
        
        
        df1 = pd.read_csv('./Firstdata/KNN_te_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teexac = df1.values
        df1 = pd.read_csv('./Firstdata/SVC_te_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teexac = df1.values
        df1 = pd.read_csv('./Firstdata/DEC_te_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teexac = df1.values
        df1 = pd.read_csv('./Firstdata/XGB_te_exac_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teexac = df1.values
        
        df1 = pd.read_csv('./Firstdata/KNN_te_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teksd = df1.values
        df1 = pd.read_csv('./Firstdata/SVC_te_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teksd = df1.values
        df1 = pd.read_csv('./Firstdata/DEC_te_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teksd = df1.values
        df1 = pd.read_csv('./Firstdata/XGB_te_KSD_count_firstdata_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teksd = df1.values
    if(dataset == 2):
        df1 = pd.read_csv('./Allstate/X_train_c_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_c = df1.values
        
        df1 = pd.read_csv('./Allstate/X_train_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_n = df1.values
        
        df1 = pd.read_csv('./Allstate/y_train_c_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_c = df1.values
        
        df1 = pd.read_csv('./Allstate/y_train_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_n = df1.values
        
        df1 = pd.read_csv('./Allstate/X_test_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        X_test = df1.values
       
        df2 = pd.read_csv('./Allstate/y_test_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        y_test = df2.values
        
        df1 = pd.read_csv('./Allstate/KNN_predict_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knnp_n = df1.values
        
        df1 = pd.read_csv('./Allstate/KNN_predict_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knnp = df1.values
        
        df1 = pd.read_csv('./Allstate/SVC_predict_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svcp_n = df1.values
        
        df1 = pd.read_csv('./Allstate/SVC_predict_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svcp = df1.values
        
        df1 = pd.read_csv('./Allstate/DEC_predict_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        decp_n = df1.values
        
        df1 = pd.read_csv('./Allstate/DEC_predict_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        decp = df1.values
        
        df1 = pd.read_csv('./Allstate/XGB_predict_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp_n = df1.values
        
        df1 = pd.read_csv('./Allstate/XGB_predict_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp = df1.values
        
        df1 = pd.read_csv('./Allstate/KNN_predictpro_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro_n = df1.values
        
        df1 = pd.read_csv('./Allstate/KNN_predictpro_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro = df1.values
        
        df1 = pd.read_csv('./Allstate/SVC_predictpro_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro_n = df1.values
        
        df1 = pd.read_csv('./Allstate/SVC_predictpro_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro = df1.values
        
        df1 = pd.read_csv('./Allstate/DEC_predictpro_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        decpro_n = df1.values
        
        df1 = pd.read_csv('./Allstate/DEC_predictpro_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        decpro = df1.values
        
        df1 = pd.read_csv('./Allstate/XGB_predictpro_n_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro_n = df1.values
        
        df1 = pd.read_csv('./Allstate/XGB_predictpro_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro = df1.values
        df1 = pd.read_csv('./Allstate/Trained test_loss_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        tra_loss = df1.values
        
        n,d = tra_loss.shape
        tra_loss_knn = tra_loss[:,0]
        tra_loss_knn = tra_loss_knn.reshape((n,1))
        tra_loss_svc = tra_loss[:,1]
        tra_loss_svc = tra_loss_svc.reshape((n,1))
        tra_loss_dec = tra_loss[:,2]
        tra_loss_dec = tra_loss_dec.reshape((n,1))
        tra_loss_xgb = tra_loss[:,3]
        tra_loss_xgb = tra_loss_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./Allstate/Trained test_NN_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        tra_nn = df1.values
        
        n,d = tra_nn.shape
        tra_nn_knn = tra_nn[:,0]
        tra_nn_knn = tra_nn_knn.reshape((n,1))
        tra_nn_svc = tra_nn[:,1]
        tra_nn_svc = tra_nn_svc.reshape((n,1))
        tra_nn_dec = tra_nn[:,2]
        tra_nn_dec = tra_nn_dec.reshape((n,1))
        tra_nn_xgb = tra_nn[:,3]
        tra_nn_xgb = tra_nn_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./Allstate/Trained test_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        tra_ksd = df1.values
        
        tra_ksd_knn = tra_ksd[:,0]
        tra_ksd_knn = tra_ksd_knn.reshape((n,1))
        tra_ksd_svc = tra_ksd[:,1]
        tra_ksd_svc = tra_ksd_svc.reshape((n,1))
        tra_ksd_dec = tra_ksd[:,2]
        tra_ksd_dec = tra_ksd_dec.reshape((n,1))
        tra_ksd_xgb = tra_ksd[:,3]
        tra_ksd_xgb = tra_ksd_xgb.reshape((n,1))
        
        df1 = pd.read_csv('./Allstate/KNN_n_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knn_exac = df1.values
        df1 = pd.read_csv('./Allstate/SVC_n_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svc_exac = df1.values
        df1 = pd.read_csv('./Allstate/DEC_n_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        dec_exac = df1.values
        df1 = pd.read_csv('./Allstate/XGB_n_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_exac = df1.values
        
        df1 = pd.read_csv('./Allstate/KNN_n_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knn_ksd = df1.values
        df1 = pd.read_csv('./Allstate/SVC_n_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svc_ksd = df1.values
        df1 = pd.read_csv('./Allstate/DEC_n_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        dec_ksd = df1.values
        df1 = pd.read_csv('./Allstate/XGB_n_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_ksd = df1.values
        
        
        df1 = pd.read_csv('./Allstate/KNN_te_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teexac = df1.values
        df1 = pd.read_csv('./Allstate/SVC_te_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teexac = df1.values
        df1 = pd.read_csv('./Allstate/DEC_te_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teexac = df1.values
        df1 = pd.read_csv('./Allstate/XGB_te_exac_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teexac = df1.values
        
        df1 = pd.read_csv('./Allstate/KNN_te_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teksd = df1.values
        df1 = pd.read_csv('./Allstate/SVC_te_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teksd = df1.values
        df1 = pd.read_csv('./Allstate/DEC_te_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teksd = df1.values
        df1 = pd.read_csv('./Allstate/XGB_te_KSD_count_allstate_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teksd = df1.values
        
    if(dataset == 3):
        df1 = pd.read_csv('./FashionMNIST/X_train_c_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_c = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/X_train_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/y_train_c_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_c = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/y_train_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/X_test_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        X_test = df1.values
       
        df2 = pd.read_csv('./FashionMNIST/y_test_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        y_test = df2.values
        
        df1 = pd.read_csv('./FashionMNIST/KNN_predict_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnp_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/KNN_predict_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnp = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/SVC_predict_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcp_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/SVC_predict_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcp = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/DEC_predict_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        decp_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/DEC_predict_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        decp = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/XGB_predict_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/XGB_predict_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/KNN_predictpro_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/KNN_predictpro_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/SVC_predictpro_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/SVC_predictpro_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/DEC_predictpro_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        decpro_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/DEC_predictpro_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        decpro = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/XGB_predictpro_n_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro_n = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/XGB_predictpro_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro = df1.values
        df1 = pd.read_csv('./FashionMNIST/Trained test_loss_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        tra_loss = df1.values
        
        n,d = tra_loss.shape
        tra_loss_knn = tra_loss[:,0]
        tra_loss_knn = tra_loss_knn.reshape((n,1))
        tra_loss_svc = tra_loss[:,1]
        tra_loss_svc = tra_loss_svc.reshape((n,1))
        tra_loss_dec = tra_loss[:,2]
        tra_loss_dec = tra_loss_dec.reshape((n,1))
        tra_loss_xgb = tra_loss[:,3]
        tra_loss_xgb = tra_loss_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./FashionMNIST/Trained test_NN_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        tra_nn = df1.values
        
        n,d = tra_nn.shape
        tra_nn_knn = tra_nn[:,0]
        tra_nn_knn = tra_nn_knn.reshape((n,1))
        tra_nn_svc = tra_nn[:,1]
        tra_nn_svc = tra_nn_svc.reshape((n,1))
        tra_nn_dec = tra_nn[:,2]
        tra_nn_dec = tra_nn_dec.reshape((n,1))
        tra_nn_xgb = tra_nn[:,3]
        tra_nn_xgb = tra_nn_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./FashionMNIST/Trained test_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        tra_ksd = df1.values
        
        tra_ksd_knn = tra_ksd[:,0]
        tra_ksd_knn = tra_ksd_knn.reshape((n,1))
        tra_ksd_svc = tra_ksd[:,1]
        tra_ksd_svc = tra_ksd_svc.reshape((n,1))
        tra_ksd_dec = tra_ksd[:,2]
        tra_ksd_dec = tra_ksd_dec.reshape((n,1))
        tra_ksd_xgb = tra_ksd[:,3]
        tra_ksd_xgb = tra_ksd_xgb.reshape((n,1))
        
        df1 = pd.read_csv('./FashionMNIST/KNN_n_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_exac = df1.values
        df1 = pd.read_csv('./FashionMNIST/SVC_n_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_exac = df1.values
        df1 = pd.read_csv('./FashionMNIST/DEC_n_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_exac = df1.values
        df1 = pd.read_csv('./FashionMNIST/XGB_n_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_exac = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/KNN_n_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_ksd = df1.values
        df1 = pd.read_csv('./FashionMNIST/SVC_n_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_ksd = df1.values
        df1 = pd.read_csv('./FashionMNIST/DEC_n_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_ksd = df1.values
        df1 = pd.read_csv('./FashionMNIST/XGB_n_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_ksd = df1.values
        
        
        df1 = pd.read_csv('./FashionMNIST/KNN_te_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teexac = df1.values
        df1 = pd.read_csv('./FashionMNIST/SVC_te_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teexac = df1.values
        df1 = pd.read_csv('./FashionMNIST/DEC_te_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teexac = df1.values
        df1 = pd.read_csv('./FashionMNIST/XGB_te_exac_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teexac = df1.values
        
        df1 = pd.read_csv('./FashionMNIST/KNN_te_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teksd = df1.values
        df1 = pd.read_csv('./FashionMNIST/SVC_te_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teksd = df1.values
        df1 = pd.read_csv('./FashionMNIST/DEC_te_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teksd = df1.values
        df1 = pd.read_csv('./FashionMNIST/XGB_te_KSD_count_fashionmnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teksd = df1.values
        
    if(dataset == 4):
        df1 = pd.read_csv('./KuzushijiMNIST/X_train_c_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_c = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/X_train_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/y_train_c_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_c = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/y_train_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/X_test_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        X_test = df1.values
       
        df2 = pd.read_csv('./KuzushijiMNIST/y_test_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        y_test = df2.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_predict_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnp_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_predict_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnp = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_predict_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcp_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_predict_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcp = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_predict_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        decp_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_predict_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        decp = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_predict_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_predict_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_predictpro_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_predictpro_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_predictpro_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_predictpro_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_predictpro_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        decpro_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_predictpro_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        decpro = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_predictpro_n_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro_n = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_predictpro_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/Trained test_loss_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        tra_loss = df1.values
        
        n,d = tra_loss.shape
        tra_loss_knn = tra_loss[:,0]
        tra_loss_knn = tra_loss_knn.reshape((n,1))
        tra_loss_svc = tra_loss[:,1]
        tra_loss_svc = tra_loss_svc.reshape((n,1))
        tra_loss_dec = tra_loss[:,2]
        tra_loss_dec = tra_loss_dec.reshape((n,1))
        tra_loss_xgb = tra_loss[:,3]
        tra_loss_xgb = tra_loss_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./KuzushijiMNIST/Trained test_NN_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        tra_nn = df1.values
        
        n,d = tra_nn.shape
        tra_nn_knn = tra_nn[:,0]
        tra_nn_knn = tra_nn_knn.reshape((n,1))
        tra_nn_svc = tra_nn[:,1]
        tra_nn_svc = tra_nn_svc.reshape((n,1))
        tra_nn_dec = tra_nn[:,2]
        tra_nn_dec = tra_nn_dec.reshape((n,1))
        tra_nn_xgb = tra_nn[:,3]
        tra_nn_xgb = tra_nn_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./KuzushijiMNIST/Trained test_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        tra_ksd = df1.values
        
        tra_ksd_knn = tra_ksd[:,0]
        tra_ksd_knn = tra_ksd_knn.reshape((n,1))
        tra_ksd_svc = tra_ksd[:,1]
        tra_ksd_svc = tra_ksd_svc.reshape((n,1))
        tra_ksd_dec = tra_ksd[:,2]
        tra_ksd_dec = tra_ksd_dec.reshape((n,1))
        tra_ksd_xgb = tra_ksd[:,3]
        tra_ksd_xgb = tra_ksd_xgb.reshape((n,1))
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_n_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_exac = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_n_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_exac = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_n_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_exac = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_n_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_exac = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_n_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_ksd = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_n_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_ksd = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_n_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_ksd = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_n_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_ksd = df1.values
        
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_te_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teexac = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_te_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teexac = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_te_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teexac = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_te_exac_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teexac = df1.values
        
        df1 = pd.read_csv('./KuzushijiMNIST/KNN_te_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teksd = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/SVC_te_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teksd = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/DEC_te_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teksd = df1.values
        df1 = pd.read_csv('./KuzushijiMNIST/XGB_te_KSD_count_kuzushijimnist_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teksd = df1.values
        
    if(dataset == 5):
        df1 = pd.read_csv('./'+str(dataname)+'/X_train_c_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_c = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/X_train_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        X_train_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/y_train_c_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_c = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/y_train_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        y_train_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/X_test_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        X_test = df1.values
       
        df2 = pd.read_csv('./'+str(dataname)+'/y_test_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        y_test = df2.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_predict_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knnp_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_predict_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knnp = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_predict_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svcp_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_predict_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svcp = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_predict_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        decp_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_predict_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        decp = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_predict_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_predict_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgbp = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_predictpro_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_predictpro_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knnpro = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_predictpro_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_predictpro_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svcpro = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_predictpro_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        decpro_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_predictpro_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        decpro = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_predictpro_n_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro_n = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_predictpro_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgbpro = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/Trained test_loss_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        tra_loss = df1.values
        
        n,d = tra_loss.shape
        tra_loss_knn = tra_loss[:,0]
        tra_loss_knn = tra_loss_knn.reshape((n,1))
        tra_loss_svc = tra_loss[:,1]
        tra_loss_svc = tra_loss_svc.reshape((n,1))
        tra_loss_dec = tra_loss[:,2]
        tra_loss_dec = tra_loss_dec.reshape((n,1))
        tra_loss_xgb = tra_loss[:,3]
        tra_loss_xgb = tra_loss_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./'+str(dataname)+'/Trained test_NN_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        tra_nn = df1.values
        
        n,d = tra_nn.shape
        tra_nn_knn = tra_nn[:,0]
        tra_nn_knn = tra_nn_knn.reshape((n,1))
        tra_nn_svc = tra_nn[:,1]
        tra_nn_svc = tra_nn_svc.reshape((n,1))
        tra_nn_dec = tra_nn[:,2]
        tra_nn_dec = tra_nn_dec.reshape((n,1))
        tra_nn_xgb = tra_nn[:,3]
        tra_nn_xgb = tra_nn_xgb.reshape((n,1))
        
        
        df1 = pd.read_csv('./'+str(dataname)+'/Trained test_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        tra_ksd = df1.values
        
        tra_ksd_knn = tra_ksd[:,0]
        tra_ksd_knn = tra_ksd_knn.reshape((n,1))
        tra_ksd_svc = tra_ksd[:,1]
        tra_ksd_svc = tra_ksd_svc.reshape((n,1))
        tra_ksd_dec = tra_ksd[:,2]
        tra_ksd_dec = tra_ksd_dec.reshape((n,1))
        tra_ksd_xgb = tra_ksd[:,3]
        tra_ksd_xgb = tra_ksd_xgb.reshape((n,1))
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_n_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knn_exac = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_n_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svc_exac = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_n_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        dec_exac = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_n_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_exac = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_n_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knn_ksd = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_n_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svc_ksd = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_n_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        dec_ksd = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_n_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_ksd = df1.values
        
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_te_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teexac = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_te_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teexac = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_te_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teexac = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_te_exac_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teexac = df1.values
        
        df1 = pd.read_csv('./'+str(dataname)+'/KNN_te_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        knn_teksd = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/SVC_te_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        svc_teksd = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/DEC_te_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        dec_teksd = df1.values
        df1 = pd.read_csv('./'+str(dataname)+'/XGB_te_KSD_count_'+str(dataname)+'_'+ str(k) +'class_state=100.csv',header = 0)
        xgb_teksd = df1.values
        
    return X_train_c, X_train_n, y_train_c, y_train_n, X_test, y_test, knnp, knnp_n, svcp, svcp_n, decp, decp_n, xgbp, xgbp_n, knnpro, knnpro_n, svcpro, svcpro_n, decpro, decpro_n, xgbpro, xgbpro_n, knn_exac, svc_exac, dec_exac, xgb_exac, knn_ksd, svc_ksd, dec_ksd, xgb_ksd, knn_teexac, svc_teexac, dec_teexac, xgb_teexac, knn_teksd, svc_teksd, dec_teksd, xgb_teksd, tra_loss_knn, tra_loss_svc, tra_loss_dec, tra_loss_xgb, tra_nn_knn, tra_nn_svc, tra_nn_dec, tra_nn_xgb, tra_ksd_knn, tra_ksd_svc, tra_ksd_dec, tra_ksd_xgb

def load_train_test_data(X,train_ratio):
    
    return sklearn.model_selection.train_test_split(X, test_size = 1 - train_ratio, random_state = 100)

def kvsall(mn_train, mn_test ,k):
    ntr,dtr = mn_train.shape
    nte,dte = mn_test.shape
    mnk_train = np.zeros((ntr,dtr))
    mnk_test = np.zeros((nte,dte))
    for i in range(ntr):
        mnk_train[i] = mn_train[i]
        if (mnk_train[i,0] == k):
            mnk_train[i,0] = 1
        else:
            mnk_train[i,0] = 0
    
    for i in range(nte):
        mnk_test[i] = mn_test[i]
        if (mnk_test[i,0] == k):
            mnk_test[i,0] = 1
        else:
            mnk_test[i,0] = 0
    
    return mnk_train, mnk_test

def buildenmen(mn_train, mn_test, mode, dataset, dataname):
    if(mode > 2):
        n,d = mn_test.shape
        aknnp = np.zeros((n,mode))
        aknn = np.zeros((n, 1))
        asvcp = np.zeros((n,mode))
        asvc = np.zeros((n, 1))
        adecp = np.zeros((n,mode))
        adec = np.zeros((n, 1))
        axgbp = np.zeros((n,mode))
        axgb = np.zeros((n, 1))
        amvp = np.zeros((n,mode))
        amv = np.zeros((n, 1))
        y_atest = mn_test[:,0].reshape(-1,1)
        for k in range(0,mode):
            print(k)
            mnk_train, mnk_test = kvsall(mn_train, mn_test, k)
            mn_train_c, mn_train_n = load_train_test_data(mnk_train,train_ratio=.6)
            X_train_c = mn_train_c[:,1:]
            X_train_n = mn_train_n[:,1:]
            y_train_c = mn_train_c[:,0].reshape(-1,1)
            y_train_n = mn_train_n[:,0].reshape(-1,1)

            X_test = mnk_test[:,1:]
            y_test = mnk_test[:,0].reshape(-1,1)
            print("Start Training")
            
            knnp_n, knnp, knnpro_n, knnpro = knn(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
            print("knn train n : %f" % (sklearn.metrics.accuracy_score(knnp_n, y_train_n)))
            print("knn test n : %f" % (sklearn.metrics.accuracy_score(knnp, y_test)))
        
            svcp_n,svcp, svcpro_n, svcpro = svc(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
            print("svc train n : %f" % (sklearn.metrics.accuracy_score(svcp_n, y_train_n)))
            print("svc test n : %f" % (sklearn.metrics.accuracy_score(svcp, y_test)))
        
            decp_n, decp, decpro_n, decpro = dectree(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
            print("dec train n : %f" % (sklearn.metrics.accuracy_score(decp_n, y_train_n)))
            print("dec test n : %f" % (sklearn.metrics.accuracy_score(decp, y_test)))
            
            xgbp_n, xgbp, xgbpro_n, xgbpro = xgboo(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
            print("xgb train n : %f" % (sklearn.metrics.accuracy_score(xgbp_n, y_train_n)))
            print("xgb test n : %f" % (sklearn.metrics.accuracy_score(xgbp, y_test)))
            
            gw = [recall_score(y_train_n, knnp_n, average='binary'), recall_score(y_train_n, svcp_n, average='binary'), recall_score(y_train_n, decp_n, average='binary'), recall_score(y_train_n, xgbp_n, average='binary')]
            print(gw)
            aknnp[:,k] = gw[0]*knnp
            asvcp[:,k] = gw[1]*svcp
            adecp[:,k] = gw[2]*decp
            axgbp[:,k] = gw[3]*xgbp
            amvp[:,k] = aknnp[:,k] + asvcp[:,k] + adecp[:,k] + axgbp[:,k]
            
            
            print("Start Counting Loss, NN and KSD")
            knnloss, svcloss, decloss, xgbloss = countloss(xgbp_n,knnpro_n,svcpro_n,decpro_n,xgbpro_n, y_train_n)
            knn_exac, svc_exac, dec_exac, xgb_exac = cacuexacc(knnp_n, svcp_n, decp_n, xgbp_n, X_train_n, y_train_n)
            knn_yte, svc_yte, dec_yte, xgb_yte = cacutestexacc(knnp_n, svcp_n, decp_n, xgbp_n, X_train_n, y_train_n, X_test)
        
            knn_nKSD, svc_nKSD, dec_nKSD, xgb_nKSD, knn_tKSD, svc_tKSD, dec_tKSD, xgb_tKSD = countKSD(X_train_n, y_train_n, X_test, y_test, knnp_n, svcp_n, decp_n, xgbp_n,knnp, svcp, decp, xgbp)
        
            print("Start Training NLoss, NNN and KSDN")
            train_loss = np.concatenate((knnloss, svcloss, decloss, xgbloss),axis = 1)
            train_NN = np.concatenate((knn_exac, svc_exac, dec_exac, xgb_exac),axis = 1)
            
            trained_loss = nn_loss(X_train_n, train_loss, X_test, k,dataset,dataname)
            trained_NN = nn_NN(X_train_n, train_NN, X_test, k,dataset,dataname)
            trained_KNNKSD = nn_KSD(X_train_n, knn_nKSD, X_test, k,dataset,dataname,1)
            trained_SVCKSD = nn_KSD(X_train_n, svc_nKSD, X_test, k,dataset,dataname,2)
            trained_DECKSD = nn_KSD(X_train_n, dec_nKSD, X_test, k,dataset,dataname,3)
            trained_XGBKSD = nn_KSD(X_train_n, xgb_nKSD, X_test, k,dataset,dataname,4)
            
            trained_KSD = np.concatenate((trained_KNNKSD, trained_SVCKSD, trained_DECKSD, trained_XGBKSD),axis = 1)
            
            savedata(X_train_c,X_train_n,y_train_c,y_train_n,X_test,y_test,knn_exac,svc_exac,dec_exac,xgb_exac,knn_yte,svc_yte,dec_yte,xgb_yte,knn_nKSD,svc_nKSD,dec_nKSD,xgb_nKSD,knn_tKSD,svc_tKSD,dec_tKSD,xgb_tKSD,knnp_n, knnp, knnpro_n, knnpro, svcp_n,svcp, svcpro_n, svcpro, decp_n, decp, decpro_n, decpro, xgbp_n, xgbp, xgbpro_n, xgbpro,trained_loss,trained_NN, trained_KSD,dataset,dataname,k)
            print("Data Saved")
        '''for i in range(n):
            aknn[i] = np.argmax(aknnp[i])
            asvc[i] = np.argmax(asvcp[i])
            adec[i] = np.argmax(adecp[i])
            axgb[i] = np.argmax(axgbp[i])
            amv[i] = np.argmax(amvp[i])
        print("knn test : %f" % (sklearn.metrics.accuracy_score(aknn, y_atest)))
        print("svc test : %f" % (sklearn.metrics.accuracy_score(asvc, y_atest)))
        print("dec test : %f" % (sklearn.metrics.accuracy_score(adec, y_atest)))
        print("xgb test : %f" % (sklearn.metrics.accuracy_score(axgb, y_atest)))
        print("mv test : %f" % (sklearn.metrics.accuracy_score(amv, y_atest)))'''
    else:
        k = str("binary")
        print(k)
        mn_train_c, mn_train_n = load_train_test_data(mn_train,train_ratio=.5)
        X_train_c = mn_train_c[:,1:]
        X_train_n = mn_train_n[:,1:]
        y_train_c = mn_train_c[:,0].reshape(-1,1)
        y_train_n = mn_train_n[:,0].reshape(-1,1)
    
        X_test = mn_test[:,1:]
        y_test = mn_test[:,0].reshape(-1,1)
        print("Start Training")
        
        knnp_n, knnp, knnpro_n, knnpro = knn(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
        print("knn train n : %f" % (sklearn.metrics.accuracy_score(knnp_n, y_train_n)))
        print("knn test n : %f" % (sklearn.metrics.accuracy_score(knnp, y_test)))
    
        svcp_n,svcp, svcpro_n, svcpro = svc(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
        print("svc train n : %f" % (sklearn.metrics.accuracy_score(svcp_n, y_train_n)))
        print("svc test n : %f" % (sklearn.metrics.accuracy_score(svcp, y_test)))

        decp_n, decp, decpro_n, decpro = dectree(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
        print("dec train n : %f" % (sklearn.metrics.accuracy_score(decp_n, y_train_n)))
        print("dec test n : %f" % (sklearn.metrics.accuracy_score(decp, y_test)))
        
        xgbp_n, xgbp, xgbpro_n, xgbpro = xgboo(X_train_c, y_train_c, X_train_n, y_train_n, X_test, k,dataset,dataname)
        print("xgb train n : %f" % (sklearn.metrics.accuracy_score(xgbp_n, y_train_n)))
        print("xgb test n : %f" % (sklearn.metrics.accuracy_score(xgbp, y_test)))
        
        mvm = mvb(knnp,svcp,decp,xgbp)
        
        '''print("knn test n : %f" % (sklearn.metrics.accuracy_score(knnp, y_test)))
        print("svc test n : %f" % (sklearn.metrics.accuracy_score(svcp, y_test)))
        print("dec test n : %f" % (sklearn.metrics.accuracy_score(decp, y_test)))
        print("xgb test n : %f" % (sklearn.metrics.accuracy_score(xgbp, y_test)))
        print("mv test n : %f" % (sklearn.metrics.accuracy_score(mvm, y_test)))'''
        
        print("Start Counting Loss, NN and KSD")
        knnloss, svcloss, decloss, xgbloss = countloss(knnpro_n,svcpro_n,decpro_n,xgbpro_n, y_train_n)
        knn_exac, svc_exac, dec_exac, xgb_exac = cacuexacc(knnp_n, svcp_n, decp_n, xgbp_n, X_train_n, y_train_n)
        knn_yte, svc_yte, dec_yte, xgb_yte = cacutestexacc(knnp_n, svcp_n, decp_n, xgbp_n, X_train_n, y_train_n, X_test)
    
        knn_nKSD, svc_nKSD, dec_nKSD, xgb_nKSD, knn_tKSD, svc_tKSD, dec_tKSD, xgb_tKSD = countKSD(X_train_n, y_train_n, X_test, y_test, knnp_n, svcp_n, decp_n, xgbp_n,knnp, svcp, decp, xgbp)
    
        print("Start Training NLoss, NNN and KSDN")
        train_loss = np.concatenate((knnloss, svcloss, decloss, xgbloss),axis = 1)
        train_NN = np.concatenate((knn_exac, svc_exac, dec_exac, xgb_exac),axis = 1)
        
        trained_loss = nn_loss(X_train_n, train_loss, X_test, k,dataset,dataname)
        trained_NN = nn_NN(X_train_n, train_NN, X_test, k,dataset,dataname)
        trained_KNNKSD = nn_KSD(X_train_n, knn_nKSD, X_test, k,dataset,dataname,1)
        trained_SVCKSD = nn_KSD(X_train_n, svc_nKSD, X_test, k,dataset,dataname,2)
        trained_DECKSD = nn_KSD(X_train_n, dec_nKSD, X_test, k,dataset,dataname,3)
        trained_XGBKSD = nn_KSD(X_train_n, xgb_nKSD, X_test, k,dataset,dataname,4)
        
        trained_KSD = np.concatenate((trained_KNNKSD, trained_SVCKSD, trained_DECKSD, trained_XGBKSD),axis = 1)
        

        savedata(X_train_c,X_train_n,y_train_c,y_train_n,X_test,y_test,knn_exac,svc_exac,dec_exac,xgb_exac,knn_yte,svc_yte,dec_yte,xgb_yte,knn_nKSD,svc_nKSD,dec_nKSD,xgb_nKSD,knn_tKSD,svc_tKSD,dec_tKSD,xgb_tKSD,knnp_n, knnp, knnpro_n, knnpro, svcp_n,svcp, svcpro_n, svcpro, decp_n, decp, decpro_n, decpro, xgbp_n, xgbp, xgbpro_n, xgbpro,trained_loss,trained_NN, trained_KSD,dataset,dataname,k)
        print("Data Saved")
    
    return knnp_n, knnp, knnpro_n, knnpro, svcp_n,svcp, svcpro_n, svcpro, decp_n, decp, decpro_n, decpro, xgbp_n, xgbp, xgbpro_n, xgbpro,trained_loss, trained_NN, trained_KSD

def knn(X_train, y_train, X_train_n, y_train_n, X_test, k,dataset,dataname):
    knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    X_train = np.concatenate((X_train,X_train_n),axis = 0)
    y_train = np.concatenate((y_train,y_train_n),axis = 0)
    knn.fit(X_train,y_train)
    
    if(dataset == 1):
        joblib.dump(knn, './Firstdata/firstdata_knn_'+str(k)+'class_state=100.pkl') 
    if(dataset == 2):
        joblib.dump(knn, './Allstate/allstate_knn_'+str(k)+'class_state=100.pkl')
    if(dataset == 3):
        joblib.dump(knn, './FashionMNIST/fashionmnist_knn_'+str(k)+'class_state=100.pkl') 
    if(dataset == 4):
        joblib.dump(knn, './KuzushijiMNIST/kuzushijimnist_knn_'+str(k)+'class_state=100.pkl')
    if(dataset == 5):
        joblib.dump(knn, './'+str(dataname) + '_knn_'+str(k)+'class_state=100.pkl') 
    
    return knn.predict(X_train_n), knn.predict(X_test), knn.predict_proba(X_train_n), knn.predict_proba(X_test)

def svc(X_train, y_train, X_train_n, y_train_n, X_test, k,dataset,dataname):
    svc = SVC(kernel='poly', max_iter= 400, random_state=None, shrinking=False, probability=True)
    X_train = np.concatenate((X_train,X_train_n),axis = 0)
    y_train = np.concatenate((y_train,y_train_n),axis = 0)
    
    print(y_train.shape)
    y_train = y_train.reshape(-1,)
    print(y_train.shape)
    
    svc.fit(X_train,y_train)
    
    if(dataset == 1):
        joblib.dump(svc, './Firstdata/firstdata_svc_'+str(k)+'class_state=100.pkl') 
    if(dataset == 2):
        joblib.dump(svc, './Allstate/allstate_svc_'+str(k)+'class_state=100.pkl')
    if(dataset == 3):
        joblib.dump(svc, './FashionMNIST/fashionmnist_svc_'+str(k)+'class_state=100.pkl') 
    if(dataset == 4):
        joblib.dump(svc, './KuzushijiMNIST/kuzushijimnist_svc_'+str(k)+'class_state=100.pkl')
    if(dataset == 5):
        joblib.dump(svc, './'+str(dataname) + '_svc_'+str(k)+'class_state=100.pkl') 
    
    return svc.predict(X_train_n), svc.predict(X_test), svc.predict_proba(X_train_n), svc.predict_proba(X_test)

def dectree(X_train, y_train, X_train_n, y_train_n, X_test, k,dataset,dataname):
    dectree = DecisionTreeClassifier(random_state=0)
    X_train = np.concatenate((X_train,X_train_n),axis = 0)
    y_train = np.concatenate((y_train,y_train_n),axis = 0)
    dectree.fit(X_train,y_train)
    
    if(dataset == 1):
        joblib.dump(dectree, './Firstdata/firstdata_dec_'+str(k)+'class_state=100.pkl') 
    if(dataset == 2):
        joblib.dump(dectree, './Allstate/allstate_dec_'+str(k)+'class_state=100.pkl')
    if(dataset == 3):
        joblib.dump(dectree, './FashionMNIST/fashionmnist_dec_'+str(k)+'class_state=100.pkl') 
    if(dataset == 4):
        joblib.dump(dectree, './KuzushijiMNIST/kuzushijimnist_dec_'+str(k)+'class_state=100.pkl')
    if(dataset == 5):
        joblib.dump(dectree, './'+str(dataname)+'/'+str(dataname)+ '_dec_'+str(k)+'class_state=100.pkl') 
    
    return dectree.predict(X_train_n), dectree.predict(X_test), dectree.predict_proba(X_train_n), dectree.predict_proba(X_test)

def xgboo(X_train, y_train, X_train_n, y_train_n, X_test, k,dataset,dataname):
    param_list = [("eta", 0.08), ("max_depth", 6), ("subsample", 0.8), ("colsample_bytree", 0.8), ("objective", "binary:logistic"), ("eval_metric", "logloss"), ("alpha", 8), ("lambda", 2)]
    n_rounds = 600
    early_stopping = 50
    
    X_train = np.concatenate((X_train,X_train_n),axis = 0)
    y_train = np.concatenate((y_train,y_train_n),axis = 0)
    
    d_train = xgb.DMatrix(X_train, label = y_train)
    d_val = xgb.DMatrix(X_train_n, label = y_train_n)
    eval_list = [(d_train, "train"), (d_val, "validation")]
    bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)
    
    if(dataset == 1):
        joblib.dump(bst, './Firstdata/firstdata_xgb_'+str(k)+'class_state=100.pkl') 
    if(dataset == 2):
        joblib.dump(bst, './Allstate/allstate_xgb_'+str(k)+'class_state=100.pkl')
    if(dataset == 3):
        joblib.dump(bst, './FashionMNIST/fashionmnist_xgb_'+str(k)+'class_state=100.pkl') 
    if(dataset == 4):
        joblib.dump(bst, './KuzushijiMNIST/kuzushijimnist_xgb_'+str(k)+'class_state=100.pkl')
    if(dataset == 5):
        joblib.dump(bst, './'+str(dataname)+'/'+str(dataname)+ '_xgb_'+str(k)+'class_state=100.pkl') 
    
    d_train_n = xgb.DMatrix(data = X_train_n)
    d_test = xgb.DMatrix(data = X_test)
    ntr, dtr =  X_train_n.shape
    nte, dte =  X_test.shape
    
    xgbpro_n = bst.predict(d_train_n)
    xgbpro = bst.predict(d_test)
    xgbp_n = np.zeros((ntr,1))
    xgbp = np.zeros((nte,1))
    
    for i in range(ntr):
        if(xgbpro_n[i] < 0.5):
            xgbp_n[i] = 0
        else:
            xgbp_n[i] = 1
    
    for i in range(nte):
        if(xgbpro[i] < 0.5):
            xgbp[i] = 0
        else:
            xgbp[i] = 1
    
    return xgbp_n, xgbp , xgbpro_n, xgbpro

def allmodel_en(y_atest,mode, dataset,dataname):
    if(mode > 2):
        print(mode)
        n,d = y_atest.shape
        aknnp = np.zeros((n,mode))
        aknn = np.zeros((n, 1))
        asvcp = np.zeros((n,mode))
        asvc = np.zeros((n, 1))
        adecp = np.zeros((n,mode))
        adec = np.zeros((n, 1))
        axgbp = np.zeros((n,mode))
        axgb = np.zeros((n, 1))
        amvp = np.zeros((n,mode))
        amv = np.zeros((n, 1))
        for i in range(0, mode):
            X_train_c, X_train_n, y_train_c, y_train_n, X_test, y_test, knnp, knnp_n, svcp, svcp_n, decp, decp_n, xgbp, xgbp_n, knnpro, knnpro_n, svcpro, svcpro_n, decpro, decpro_n, xgbpro, xgbpro_n, knn_exac, svc_exac, dec_exac, xgb_exac, knn_ksd, svc_ksd, dec_ksd, xgb_ksd, knn_teexac, svc_teexac, dec_teexac, xgb_teexac, knn_teksd, svc_teksd, dec_teksd, xgb_teksd, tra_loss_knn, tra_loss_svc, tra_loss_dec, tra_loss_xgb, tra_nn_knn, tra_nn_svc, tra_nn_dec, tra_nn_xgb, tra_ksd_knn, tra_ksd_svc, tra_ksd_dec, tra_ksd_xgb = readdata_check(i,dataset,dataname)
            n,d = y_test.shape
            en_m1_1_tmp = np.zeros((n,mode))
            en_m1_1 = np.zeros((n,1))
            ksd_m1_1 = np.ones((n,1))
            en_m1_2_tmp = np.zeros((n,mode))
            enm1_2p_reg = np.zeros((n,1))
            en_m1_2 = np.zeros((n,1))
    
            en_m2_1_tmp = np.zeros((n,mode))
            en_m2_1 = np.zeros((n,1))
            ksd_m2_1 = np.ones((n,1))
            en_m2_2_tmp = np.zeros((n,mode))
            en_m2_2 = np.zeros((n,1))
            ksd_m2_2 = np.ones((n,1))
    
            en_m3_1_tmp = np.zeros((n,mode))
            en_m3_1 = np.zeros((n,1))
            ksd_m3_1 = np.ones((n,1))
            en_m3_2_tmp = np.zeros((n,mode))
            en_m3_2 = np.zeros((n,1))
            ksd_m3_2 = np.ones((n,1))
    
            en_m4_1_tmp = np.zeros((n,mode))
            en_m4_1 = np.zeros((n,1))
            ksd_m4_1 = np.ones((n,1))
            en_m4_2_tmp = np.zeros((n,mode))
            en_m4_2 = np.zeros((n,1))
            ksd_m4_2 = np.ones((n,1))
    
            en_m5_1_tmp = np.zeros((n,mode))
            en_m5_1 = np.zeros((n,1))
            en_m5_2_tmp = np.zeros((n,mode))
            en_m5_2 = np.zeros((n,1))
            en_m5_3_tmp = np.zeros((n,mode))
            en_m5_3 = np.zeros((n,1))
            en_m5_4_tmp = np.zeros((n,mode))
            en_m5_4 = np.zeros((n,1))
    
            en_mm1_tmp = np.zeros((n,mode))
            en_mm1 = np.zeros((n,1))
            en_mm2_tmp = np.zeros((n,mode))
            en_mm2 = np.zeros((n,1))
            
            y = np.concatenate((knnp, svcp, decp, xgbp),axis = 1)
            gw = [recall_score(y_train_n, knnp_n, average='binary'), recall_score(y_train_n, svcp_n, average='binary'), recall_score(y_train_n, decp_n, average='binary'), recall_score(y_train_n, xgbp_n, average='binary')]
            print(gw)
            
            nn_true = np.concatenate((knn_teexac, svc_teexac, dec_teexac, xgb_teexac),axis = 1)
            ksd_true = np.concatenate((knn_teksd, svc_teksd, dec_teksd, xgb_teksd),axis = 1)
            trained_loss = np.concatenate((tra_loss_knn, tra_loss_svc, tra_loss_dec, tra_loss_xgb),axis = 1)
            trained_NN = np.concatenate((tra_nn_knn, tra_nn_svc, tra_nn_dec, tra_nn_xgb),axis = 1)
            trained_KSD = np.concatenate((tra_ksd_knn, tra_ksd_svc, tra_ksd_dec, tra_ksd_xgb),axis = 1)
            #print(sklearn.metrics.r2_score(nn_true, trained_NN))
            #print(sklearn.metrics.r2_score(ksd_true, trained_KSD))
            
            aknnp[:,i] = gw[0]*knnp
            asvcp[:,i] = gw[1]*svcp
            adecp[:,i] = gw[2]*decp
            axgbp[:,i] = gw[3]*xgbp
            amvp[:,i] = aknnp[:,i] + asvcp[:,i] + adecp[:,i] + axgbp[:,i]
            
            
            weight_m1_1 = np.zeros((n,4))

            for j in range(n):
                weight_m1_1[j,0] = knnpro[j,int(knnp[j])]
                weight_m1_1[j,1] = svcpro[j,int(svcp[j])]
                weight_m1_1[j,2] = decpro[j,int(decp[j])]

                if(xgbp[j] == 1):
                    weight_m1_1[j,3] = xgbpro[j]
                else:
                    weight_m1_1[j,3] = 1 - xgbpro[j]

            print(weight_m1_1.shape, y.shape, ksd_m1_1.shape)
            enm1_1p_reg = en(weight_m1_1, y, ksd_m1_1, gw, 0)
            enm1_1p_reg = enm1_1p_reg.reshape((n,))
            en_m1_1_tmp[:,i] = enm1_1p_reg

            #model 1-2
            for j in range(n):
                p1 = knnpro[j,1] + svcpro[j,1] + decpro[j,1] + xgbpro[j]
                p0 = knnpro[j,0] + svcpro[j,0] + decpro[j,0] + (1 - xgbpro[j])

                if(p1 > p0):
                    enm1_2p_reg[j] = 1
                else:
                    enm1_2p_reg[j] = 0

            enm1_2p_reg = enm1_2p_reg.reshape((n,))
            en_m1_2_tmp[:,i] = enm1_2p_reg

            #model 2-1
            weight_m2_1 = 1/trained_loss
            enm2_1p_reg = en(weight_m2_1, y, ksd_m2_1, gw, 0)
            enm2_1p_reg = enm2_1p_reg.reshape((n,))
            en_m2_1_tmp[:,i] = enm2_1p_reg

            #model 2-2
            weight_m2_2 = 1 - trained_loss
            enm2_2p_reg = en(weight_m2_2, y, ksd_m2_2, gw, 0)
            enm2_2p_reg = enm2_2p_reg.reshape((n,))
            en_m2_2_tmp[:,i] = enm2_2p_reg


            #model 3-1
            weight_m3_1 = trained_NN/30
            enm3_1p_reg = en(weight_m3_1, y, ksd_m3_1, gw, 0)
            enm3_1p_reg = enm3_1p_reg.reshape((n,))
            en_m3_1_tmp[:,i] = enm3_1p_reg

            #model 3-2
            weight_m3_2 = trained_NN/30
            enm3_2p_reg = en(weight_m3_2, y, ksd_m3_2, gw, 0)
            enm3_2p_reg = enm3_2p_reg.reshape((n,))
            en_m3_2_tmp[:,i] = enm3_2p_reg

            #model 4-1
            weight_m4_1 = trained_NN
            enm4_1p_reg = en(weight_m4_1, y, ksd_m4_1, gw, 1)
            enm4_1p_reg = enm4_1p_reg.reshape((n,))
            en_m4_1_tmp[:,i] = enm4_1p_reg

            #model 4-2
            weight_m4_2 = trained_NN
            enm4_2p_reg = en(weight_m4_2, y, ksd_m4_2, gw, 1)
            enm4_2p_reg = enm4_2p_reg.reshape((n,))
            en_m4_2_tmp[:,i] = enm4_2p_reg


            #model 5-1
            weight_m5_1 = trained_NN/30
            ksd_m5_1 = trained_KSD
            enm5_1p_reg = en(weight_m5_1, y , ksd_m5_1, gw, 0)
            enm5_1p_reg = enm5_1p_reg.reshape((n,))
            en_m5_1_tmp[:,i] = enm5_1p_reg

            #model 5-2
            weight_m5_2 = trained_NN
            ksd_m5_2 = trained_KSD
            enm5_2p_reg = en(weight_m5_2, y, ksd_m5_2, gw, 0)
            enm5_2p_reg = enm5_2p_reg.reshape((n,))
            en_m5_2_tmp[:,i] = enm5_2p_reg

            #model 5-3
            weight_m5_3 = trained_NN/30
            ksd_m5_3 = trained_KSD
            enm5_3p_reg = en(weight_m5_3, y , ksd_m5_3, gw, 1)
            enm5_3p_reg = enm5_3p_reg.reshape((n,))
            en_m5_3_tmp[:,i] = enm5_3p_reg

            #model 5-4
            weight_m5_4 = trained_NN
            ksd_m5_4 = trained_KSD
            enm5_4p_reg = en(weight_m5_4, y, ksd_m5_4, gw, 1)
            enm5_4p_reg = enm5_4p_reg.reshape((n,))
            en_m5_4_tmp[:,i] = enm5_4p_reg

            #理想model
            weight_mm = nn_true
            ksd_mm = ksd_true
            enmm1p_reg = en(weight_mm, y, ksd_mm, gw, 0)
            enmm1p_reg = enmm1p_reg.reshape((n,))
            en_mm1_tmp[:,i] = enmm1p_reg

            enmm2p_reg = en(weight_mm, y, ksd_mm, gw, 1)
            enmm2p_reg = enmm2p_reg.reshape((n,))
            en_mm2_tmp[:,i] = enmm2p_reg


        for i in range(n):
            en_m1_1[i] = np.argmax(en_m1_1_tmp[i])
            en_m1_2[i] = np.argmax(en_m1_2_tmp[i])
            en_m2_1[i] = np.argmax(en_m2_1_tmp[i])
            en_m2_2[i] = np.argmax(en_m2_2_tmp[i])
            en_m3_1[i] = np.argmax(en_m3_1_tmp[i])
            en_m3_2[i] = np.argmax(en_m3_2_tmp[i])
            en_m4_1[i] = np.argmax(en_m4_1_tmp[i])
            en_m4_2[i] = np.argmax(en_m4_2_tmp[i])
            en_m5_1[i] = np.argmax(en_m5_1_tmp[i])
            en_m5_2[i] = np.argmax(en_m5_2_tmp[i])
            en_m5_3[i] = np.argmax(en_m5_3_tmp[i])
            en_m5_4[i] = np.argmax(en_m5_4_tmp[i])

            en_mm1[i] = np.argmax(en_mm1_tmp[i])
            en_mm2[i] = np.argmax(en_mm2_tmp[i])
            
            aknn[i] = np.argmax(aknnp[i])
            asvc[i] = np.argmax(asvcp[i])
            adec[i] = np.argmax(adecp[i])
            axgb[i] = np.argmax(axgbp[i])
            amv[i] = np.argmax(amvp[i])
        print("knn test : %f" % (sklearn.metrics.accuracy_score(aknn, y_atest)))
        print("svc test : %f" % (sklearn.metrics.accuracy_score(asvc, y_atest)))
        print("dec test : %f" % (sklearn.metrics.accuracy_score(adec, y_atest)))
        print("xgb test : %f" % (sklearn.metrics.accuracy_score(axgb, y_atest)))
        print("mv test : %f" % (sklearn.metrics.accuracy_score(amv, y_atest)))
    
    else:
        k = str("binary")
        print(k)
        X_train_c, X_train_n, y_train_c, y_train_n, X_test, y_test, knnp, knnp_n, svcp, svcp_n, decp, decp_n, xgbp, xgbp_n, knnpro, knnpro_n, svcpro, svcpro_n, decpro, decpro_n, xgbpro, xgbpro_n, knn_exac, svc_exac, dec_exac, xgb_exac, knn_ksd, svc_ksd, dec_ksd, xgb_ksd, knn_teexac, svc_teexac, dec_teexac, xgb_teexac, knn_teksd, svc_teksd, dec_teksd, xgb_teksd, tra_loss_knn, tra_loss_svc, tra_loss_dec, tra_loss_xgb, tra_nn_knn, tra_nn_svc, tra_nn_dec, tra_nn_xgb, tra_ksd_knn, tra_ksd_svc, tra_ksd_dec, tra_ksd_xgb = readdata_check(k,dataset,dataname)
        n,d = y_atest.shape
        en_m1_1_tmp = np.zeros((n,1))
        en_m1_1 = np.zeros((n,1))
        ksd_m1_1 = np.ones((n,1))
        en_m1_2_tmp = np.zeros((n,1))
        enm1_2p_reg = np.zeros((n,1))
        en_m1_2 = np.zeros((n,1))
    
        en_m2_1_tmp = np.zeros((n,1))
        en_m2_1 = np.zeros((n,1))
        ksd_m2_1 = np.ones((n,1))
        en_m2_2_tmp = np.zeros((n,1))
        en_m2_2 = np.zeros((n,1))
        ksd_m2_2 = np.ones((n,1))
    
        en_m3_1_tmp = np.zeros((n,1))
        en_m3_1 = np.zeros((n,1))
        ksd_m3_1 = np.ones((n,1))
        en_m3_2_tmp = np.zeros((n,1))
        en_m3_2 = np.zeros((n,1))
        ksd_m3_2 = np.ones((n,1))
    
        en_m4_1_tmp = np.zeros((n,1))
        en_m4_1 = np.zeros((n,1))
        ksd_m4_1 = np.ones((n,1))
        en_m4_2_tmp = np.zeros((n,1))
        en_m4_2 = np.zeros((n,1))
        ksd_m4_2 = np.ones((n,1))
    
        en_m5_1_tmp = np.zeros((n,1))
        en_m5_1 = np.zeros((n,1))
        en_m5_2_tmp = np.zeros((n,1))
        en_m5_2 = np.zeros((n,1))
        en_m5_3_tmp = np.zeros((n,1))
        en_m5_3 = np.zeros((n,1))
        en_m5_4_tmp = np.zeros((n,1))
        en_m5_4 = np.zeros((n,1))
    
        en_mm1_tmp = np.zeros((n,1))
        en_mm1 = np.zeros((n,1))
        en_mm2_tmp = np.zeros((n,1))
        en_mm2 = np.zeros((n,1))
        y = np.concatenate((knnp, svcp, decp, xgbp),axis = 1)
        gw = [sklearn.metrics.accuracy_score(y_train_n, knnp_n), sklearn.metrics.accuracy_score(y_train_n, svcp_n), sklearn.metrics.accuracy_score(y_train_n, decp_n), sklearn.metrics.accuracy_score(y_train_n, xgbp_n)]#
        print(gw)
        
        nn_true = np.concatenate((knn_teexac, svc_teexac, dec_teexac, xgb_teexac),axis = 1)
        ksd_true = np.concatenate((knn_teksd, svc_teksd, dec_teksd, xgb_teksd),axis = 1)
        trained_loss = np.concatenate((tra_loss_knn, tra_loss_svc, tra_loss_dec, tra_loss_xgb),axis = 1)
        trained_NN = np.concatenate((tra_nn_knn, tra_nn_svc, tra_nn_dec, tra_nn_xgb),axis = 1)
        trained_KSD = np.concatenate((tra_ksd_knn, tra_ksd_svc, tra_ksd_dec, tra_ksd_xgb),axis = 1)
        #print(sklearn.metrics.r2_score(nn_true, trained_NN))
        #print(sklearn.metrics.r2_score(ksd_true, trained_KSD))
        mvm = mvb(knnp,svcp,decp,xgbp)
        
        #model 1-1
        weight_m1_1 = np.zeros((n,4))
    
        for j in range(n):
            weight_m1_1[j,0] = knnpro[j,int(knnp[j])]
            weight_m1_1[j,1] = svcpro[j,int(svcp[j])]
            weight_m1_1[j,2] = decpro[j,int(decp[j])]
            
            if(xgbp[j] == 1):
                weight_m1_1[j,3] = xgbpro[j]
            else:
                weight_m1_1[j,3] = 1 - xgbpro[j]
    
        #print(weight_m1_1.shape, y.shape, ksd_m1_1.shape)
        enm1_1p_reg = en(weight_m1_1, y, ksd_m1_1, gw, 0)
        enm1_1p_reg = enm1_1p_reg.reshape((n,))
        en_m1_1_tmp = enm1_1p_reg
    
        #model 1-2
        for j in range(n):
            p1 = knnpro[j,1] + svcpro[j,1] + decpro[j,1] + xgbpro[j]
            p0 = knnpro[j,0] + svcpro[j,0] + decpro[j,0] + (1 - xgbpro[j])
    
            if(p1 > p0):
                enm1_2p_reg[j] = 1
            else:
                enm1_2p_reg[j] = 0
    
        enm1_2p_reg = enm1_2p_reg.reshape((n,))
        en_m1_2_tmp = enm1_2p_reg
    
        #model 2-1
        weight_m2_1 = 1/trained_loss
        enm2_1p_reg = en(weight_m2_1, y, ksd_m2_1, gw, 0)
        enm2_1p_reg = enm2_1p_reg.reshape((n,))
        en_m2_1_tmp = enm2_1p_reg
    
        #model 2-2
        weight_m2_2 = 1 - trained_loss
        enm2_2p_reg = en(weight_m2_2, y, ksd_m2_2, gw, 0)
        enm2_2p_reg = enm2_2p_reg.reshape((n,))
        en_m2_2_tmp = enm2_2p_reg
    
    
        #model 3-1
        weight_m3_1 = trained_NN/30
        enm3_1p_reg = en(weight_m3_1, y, ksd_m3_1, gw, 0)
        enm3_1p_reg = enm3_1p_reg.reshape((n,))
        en_m3_1_tmp = enm3_1p_reg
    
        #model 3-2
        weight_m3_2 = trained_NN/30
        enm3_2p_reg = en(weight_m3_2, y, ksd_m3_2, gw, 1)
        enm3_2p_reg = enm3_2p_reg.reshape((n,))
        en_m3_2_tmp = enm3_2p_reg
    
        #model 4-1
        weight_m4_1 = trained_NN
        enm4_1p_reg = en(weight_m4_1, y, ksd_m4_1, gw, 0)
        enm4_1p_reg = enm4_1p_reg.reshape((n,))
        en_m4_1_tmp = enm4_1p_reg
    
        #model 4-2
        weight_m4_2 = trained_NN
        enm4_2p_reg = en(weight_m4_2, y, ksd_m4_2, gw, 1)
        enm4_2p_reg = enm4_2p_reg.reshape((n,))
        en_m4_2_tmp = enm4_2p_reg
    
    
        #model 5-1
        weight_m5_1 = trained_NN/30
        ksd_m5_1 = trained_KSD
        enm5_1p_reg = en(weight_m5_1, y , ksd_m5_1, gw, 0)
        enm5_1p_reg = enm5_1p_reg.reshape((n,))
        en_m5_1_tmp = enm5_1p_reg
    
        #model 5-2
        weight_m5_2 = trained_NN
        ksd_m5_2 = trained_KSD
        enm5_2p_reg = en(weight_m5_2, y, ksd_m5_2, gw, 0)
        enm5_2p_reg = enm5_2p_reg.reshape((n,))
        en_m5_2_tmp = enm5_2p_reg
    
        #model 5-3
        weight_m5_3 = trained_NN/30
        ksd_m5_3 = trained_KSD
        enm5_3p_reg = en(weight_m5_3, y , ksd_m5_3, gw, 1)
        enm5_3p_reg = enm5_3p_reg.reshape((n,))
        en_m5_3_tmp = enm5_3p_reg
    
        #model 5-4
        weight_m5_4 = trained_NN/30
        ksd_m5_4 = trained_KSD
        enm5_4p_reg = en(weight_m5_4, y, ksd_m5_4, gw, 1)
        enm5_4p_reg = enm5_4p_reg.reshape((n,))
        en_m5_4_tmp = enm5_4p_reg
    
        #理想model
        weight_mm = nn_true
        ksd_mm = ksd_true
        enmm1p_reg = en(weight_mm, y, ksd_mm, gw, 0)
        enmm1p_reg = enmm1p_reg.reshape((n,))
        en_mm1_tmp = enmm1p_reg
    
        enmm2p_reg = en(weight_mm, y, ksd_mm, gw, 1)
        enmm2p_reg = enmm2p_reg.reshape((n,))
        en_mm2_tmp = enmm2p_reg
    
    
        for i in range(n):
            if(en_m1_1_tmp[i] > 0.5):
                en_m1_1[i] = 1
            else:
                en_m1_1[i] = 0
                
            if(en_m1_2_tmp[i] > 0.5):
                en_m1_2[i] = 1
            else:
                en_m1_2[i] = 0
                
            if(en_m2_1_tmp[i] > 0.5):
                en_m2_1[i] = 1
            else:
                en_m2_1[i] = 0
                
            if(en_m2_2_tmp[i] > 0.5):
                en_m2_2[i] = 1
            else:
                en_m2_2[i] = 0
            
            if(en_m3_1_tmp[i] > 0.5):
                en_m3_1[i] = 1
            else:
                en_m3_1[i] = 0
            
            if(en_m3_2_tmp[i] > 0.5):
                en_m3_2[i] = 1
            else:
                en_m3_2[i] = 0
            
            if(en_m4_1_tmp[i] > 0.5):
                en_m4_1[i] = 1
            else:
                en_m4_1[i] = 0
            
            if(en_m4_2_tmp[i] > 0.5):
                en_m4_2[i] = 1
            else:
                en_m4_2[i] = 0
            
            if(en_m5_1_tmp[i] > 0.5):
                en_m5_1[i] = 1
            else:
                en_m5_1[i] = 0
            
            if(en_m5_2_tmp[i] > 0.5):
                en_m5_2[i] = 1
            else:
                en_m5_2[i] = 0
            
            if(en_m5_3_tmp[i] > 0.5):
                en_m5_3[i] = 1
            else:
                en_m5_3[i] = 0
                
            if(en_m5_4_tmp[i] > 0.5):
                en_m5_4[i] = 1
            else:
                en_m5_4[i] = 0
            
            
            if(en_mm1_tmp[i] > 0.5):
                en_mm1[i] = 1
            else:
                en_mm1[i] = 0
                
            
            if(en_mm2_tmp[i] > 0.5):
                en_mm2[i] = 1
            else:
                en_mm2[i] = 0
            
        print("knn test : %f" % (sklearn.metrics.accuracy_score(knnp, y_atest)))
        print("svc test : %f" % (sklearn.metrics.accuracy_score(svcp, y_atest)))
        print("dec test : %f" % (sklearn.metrics.accuracy_score(decp, y_atest)))
        print("xgb test : %f" % (sklearn.metrics.accuracy_score(xgbp, y_atest)))
        print("mv test : %f" % (sklearn.metrics.accuracy_score(mvm, y_atest)))
    print("m1-1 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m1_1, y_atest)))
    print("m1-2 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m1_2, y_atest)))
    print("m2-1 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m2_1, y_atest)))
    print("m2-2 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m2_2, y_atest)))
    print("m3-1 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m3_1, y_atest)))
    print("m3-2 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m3_2, y_atest)))
    print("m4-1 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m4_1, y_atest)))
    print("m4-2 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m4_2, y_atest)))
    print("m5-1 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m5_1, y_atest)))
    print("m5-2 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m5_2, y_atest)))
    print("m5-3 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m5_3, y_atest)))
    print("m5-4 ensemble test : %f" % (sklearn.metrics.accuracy_score(en_m5_4, y_atest)))
    
    print("理想 ensemble without softmax test : %f" % (sklearn.metrics.accuracy_score(en_mm1, y_atest)))
    print("理想 ensemble with softmax test : %f" % (sklearn.metrics.accuracy_score(en_mm2, y_atest)))

    return 0

def countloss(class1_prtrpro_n, class2_prtrpro_n, class3_prtrpro_n, class4_prtrpro_n,y_train_n):
    n,d = y_train_n.shape
    cl1loss = np.zeros((n,1))
    cl2loss = np.zeros((n,1))
    cl3loss = np.zeros((n,1))
    cl4loss = np.zeros((n,1))
    
    for i in range(n):
        if(y_train_n[i] == 1):
            cl1loss[i] = 1 - class1_prtrpro_n[i,1]
            cl2loss[i] = 1 - class2_prtrpro_n[i,1]
            cl3loss[i] = 1 - class3_prtrpro_n[i,1]
            cl4loss[i] = 1 - class4_prtrpro_n[i]
            
        if(y_train_n[i] == 0):
            cl1loss[i] = abs(0 - class1_prtrpro_n[i,0])
            cl2loss[i] = abs(0 - class2_prtrpro_n[i,0])
            cl3loss[i] = abs(0 - class3_prtrpro_n[i,0])
            cl4loss[i] = abs(0 - (1 - class4_prtrpro_n[i]))
    
    return cl1loss,cl2loss,cl3loss,cl4loss

def cacuexacc(class1_prtr_n, class2_prtr_n, class3_prtr_n, class4_prtr_n, X_train_n, y_train_n):
    n,d = X_train_n.shape
    exac1 = np.zeros((n,1))
    exac2 = np.zeros((n,1))
    exac3 = np.zeros((n,1))
    exac4 = np.zeros((n,1))
    
    for i in range (n):
        cor1 = 0
        cor2 = 0
        cor3 = 0
        cor4 = 0
        #total = 0
        k = 30
        dis_index = np.zeros((n))
        #print(dis_index.shape)
        for j in range(n):
            a = LA.norm(X_train_n[i] - X_train_n[j])
            #print(a)
            dis_index[j] = a
        
        #print(dis_index)
        b = np.argsort(dis_index)
        #print(b)
        
        for l in range(k):
            #print(b[k])
            if(class1_prtr_n[b[l]] == y_train_n[b[l]]):
                cor1 = cor1 + 1
            if(class2_prtr_n[b[l]] == y_train_n[b[l]]):
                cor2 = cor2 + 1
            if(class3_prtr_n[b[l]] == y_train_n[b[l]]):
                cor3 = cor3 + 1
            if(class4_prtr_n[b[l]] == y_train_n[b[l]]):
                cor4 = cor4 + 1
                
        #print(cor)
        #print(k)
        exac1[i] = cor1
        exac2[i] = cor2
        exac3[i] = cor3
        exac4[i] = cor4
        #print(i)
        #print(exac1[i],exac2[i],exac3[i])
        
        
    #print(exac.shape)
    return exac1, exac2, exac3, exac4

def cacutestexacc(class1_prtr_n, class2_prtr_n, class3_prtr_n,class4_prtr_n, X_train_n, y_train_n, X_test):
    ne,de = X_test.shape
    nr,dr = X_train_n.shape
    exac1 = np.zeros((ne,1))
    exac2 = np.zeros((ne,1))
    exac3 = np.zeros((ne,1))
    exac4 = np.zeros((ne,1))
    
    for i in range (ne):
        cor1 = 0
        cor2 = 0
        cor3 = 0
        cor4 = 0
        #total = 0
        k = 30
        dis_index = np.zeros((nr))
        #print(dis_index.shape)
        for j in range(nr):
            a = LA.norm(X_test[i] - X_train_n[j])
            dis_index[j] = a
        
        #print(dis_index)
        b = np.argsort(dis_index)
        #print(b)
        
        for l in range(k):
            #print(b[k])
            if(class1_prtr_n[b[l]] == y_train_n[b[l]]):
                cor1 = cor1 + 1
            if(class2_prtr_n[b[l]] == y_train_n[b[l]]):
                cor2 = cor2 + 1
            if(class3_prtr_n[b[l]] == y_train_n[b[l]]):
                cor3 = cor3 + 1
            if(class4_prtr_n[b[l]] == y_train_n[b[l]]):
                cor4 = cor4 + 1
                
        #print(cor)
        #print(k)
        #print(acc)
        exac1[i] = cor1
        exac2[i] = cor2
        exac3[i] = cor3
        exac4[i] = cor4
        #print(i)
        #print(exac1[i],exac2[i],exac3[i])
        
        
    #print(exac.shape)
    return exac1, exac2, exac3,exac4

def countKSD(X_train_n, y_train_n, X_test, y_test, knnp_n, svcp_n, decp_n, xgbp_n, knnp, svcp, decp, xgbp):
    nt,dt = X_train_n.shape
    nte,dte = X_test.shape
    #count train n KSD
    train_nknn_index = []
    train_nsvc_index = []
    train_ndec_index = []
    train_nxgb_index = []
    
    for i in range(nt):
        if(knnp_n[i]==y_train_n[i]):
            train_nknn_index.append(i)
        if(svcp_n[i]==y_train_n[i]):
            train_nsvc_index.append(i)
        if(decp_n[i]==y_train_n[i]):
            train_ndec_index.append(i)
        if(xgbp_n[i]==y_train_n[i]):
            train_nxgb_index.append(i)
            #print('++')
            
    train_nknn_index = np.sort(train_nknn_index)
    train_nsvc_index = np.sort(train_nsvc_index)
    train_ndec_index = np.sort(train_ndec_index)
    train_nxgb_index = np.sort(train_nxgb_index)
    print(len(train_nxgb_index))
    
    train_nknn_K = np.zeros((nt,1))
    train_nsvc_S = np.zeros((nt,1))
    train_ndec_D = np.zeros((nt,1))
    train_nxgb_X = np.zeros((nt,1))
    
    for i in range(len(train_nknn_index)):
        train_nknn_K[train_nknn_index[i]] = 1
    for j in range(len(train_nsvc_index)):
        train_nsvc_S[train_nsvc_index[j]] = 1
    for k in range(len(train_ndec_index)):
        train_ndec_D[train_ndec_index[k]] = 1
    for l in range(len(train_nxgb_index)):
        train_nxgb_X[train_nxgb_index[l]] = 1
    
    #count te KSD
    train_tknn_index = []
    train_tsvc_index = []
    train_tdec_index = []
    train_txgb_index = []
    
    for i in range(nte):
        if(knnp[i]==y_test[i]):
            train_tknn_index.append(i)
        if(svcp[i]==y_test[i]):
            train_tsvc_index.append(i)
        if(decp[i]==y_test[i]):
            train_tdec_index.append(i)
        if(xgbp[i]==y_test[i]):
            train_txgb_index.append(i)
            #print("++")
            
    train_tknn_index = np.sort(train_tknn_index)
    train_tsvc_index = np.sort(train_tsvc_index)
    train_tdec_index = np.sort(train_tdec_index)
    train_txgb_index = np.sort(train_txgb_index)
    print(len(train_txgb_index))
    
    train_tknn_K = np.zeros((nte,1))
    train_tsvc_S = np.zeros((nte,1))
    train_tdec_D = np.zeros((nte,1))
    train_txgb_X = np.zeros((nte,1))
    
    for i in range(len(train_tknn_index)):
        train_tknn_K[train_tknn_index[i]] = 1
    for j in range(len(train_tsvc_index)):
        train_tsvc_S[train_tsvc_index[j]] = 1
    for k in range(len(train_tdec_index)):
        train_tdec_D[train_tdec_index[k]] = 1
    for l in range(len(train_txgb_index)):
        train_txgb_X[train_txgb_index[l]] = 1
        
    return train_nknn_K, train_nsvc_S, train_ndec_D, train_nxgb_X, train_tknn_K, train_tsvc_S, train_tdec_D, train_txgb_X

def nn_KSD(X,y,X_t,k,dataset,dataname,a):
    n,d = y.shape
    #for i in range(d):
    #nte, dte = X_t.shape
    #y_hat = np.zeros((nte, d))
    param_list = [("eta", 0.08), ("max_depth", 6), ("subsample", 0.8), ("colsample_bytree", 0.8), ("objective", "binary:logistic"), ("eval_metric", "rmse"), ("alpha", 8), ("lambda", 2)]
    n_rounds = 500
    early_stopping = 50
    #a = y[:,i].reshape(-1,1)
    #print(a.shape)
    d_train = xgb.DMatrix(X, label = y)
    #d_val = xgb.DMatrix(X_train_n, label = y_train_n)
    eval_list = [(d_train, "train")]
    bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)
    #print(y.shape)
    d_test = xgb.DMatrix(X_t)
    
    if(a == 1):
        if(dataset == 1):
            joblib.dump(bst, './Firstdata/firstdata_k_'+str(k)+'class_state=100.pkl')
        if(dataset == 2):
            joblib.dump(bst, './Allstate/allstate_k_'+str(k)+'class_state=100.pkl')
        if(dataset == 3):
            joblib.dump(bst, './FashionMNIST/fashionmnist_k_'+str(k)+'class_state=100.pkl')
        if(dataset == 4):
            joblib.dump(bst, './KuzushijiMNIST/kuzushijimnist_k_'+str(k)+'class_state=100.pkl')
        if(dataset == 5):
            joblib.dump(bst, './'+str(dataname)+'/'+str(dataname)+'_k_'+str(k)+'class_state=100.pkl')
            
    if(a == 2):
        if(dataset == 1):
            joblib.dump(bst, './Firstdata/firstdata_s_'+str(k)+'class_state=100.pkl')
        if(dataset == 2):
            joblib.dump(bst, './Allstate/allstate_s_'+str(k)+'class_state=100.pkl')
        if(dataset == 3):
            joblib.dump(bst, './FashionMNIST/fashionmnist_s_'+str(k)+'class_state=100.pkl')
        if(dataset == 4):
            joblib.dump(bst, './KuzushijiMNIST/kuzushijimnist_s_'+str(k)+'class_state=100.pkl')
        if(dataset == 5):
            joblib.dump(bst, './'+str(dataname)+'/'+str(dataname)+'_s_'+str(k)+'class_state=100.pkl')
    
    if(a == 3):
        if(dataset == 1):
            joblib.dump(bst, './Firstdata/firstdata_d_'+str(k)+'class_state=100.pkl')
        if(dataset == 2):
            joblib.dump(bst, './Allstate/allstate_d_'+str(k)+'class_state=100.pkl')
        if(dataset == 3):
            joblib.dump(bst, './FashionMNIST/fashionmnist_d_'+str(k)+'class_state=100.pkl')
        if(dataset == 4):
            joblib.dump(bst, './KuzushijiMNIST/kuzushijimnist_d_'+str(k)+'class_state=100.pkl')
        if(dataset == 5):
            joblib.dump(bst, './'+str(dataname)+'/'+str(dataname)+'_d_'+str(k)+'class_state=100.pkl')
    
    if(a == 4):
        if(dataset == 1):
            joblib.dump(bst, './Firstdata/firstdata_x_'+str(k)+'class_state=100.pkl')
        if(dataset == 2):
            joblib.dump(bst, './Allstate/allstate_x_'+str(k)+'class_state=100.pkl')
        if(dataset == 3):
            joblib.dump(bst, './FashionMNIST/fashionmnist_x_'+str(k)+'class_state=100.pkl')
        if(dataset == 4):
            joblib.dump(bst, './KuzushijiMNIST/kuzushijimnist_x_'+str(k)+'class_state=100.pkl')
        if(dataset == 5):
            joblib.dump(bst, './'+str(dataname)+'/'+str(dataname)+'_x_'+str(k)+'class_state=100.pkl')
    
    
    return bst.predict(d_test).reshape(-1,1)

def nn_NN(X,y,X_t,k,dataset,dataname):
    
    if(dataset == 2):
        n,d = X.shape
        xs = tf.placeholder(tf.float32, [None, d])
        ys = tf.placeholder(tf.float32, [None, 4])
        
        l1 = tf.layers.dense(inputs = xs, units = n, activation = tf.nn.relu)
        l2 = tf.layers.dense(inputs = l1, units = 770, activation = tf.nn.relu)
        l3 = tf.layers.dense(inputs = l2, units = 386, activation = tf.nn.relu)
        prediction = tf.layers.dense(inputs = l3, units = 4, activation = None)
        
        x_data = X.astype(np.float32)
        y_data = y.astype(np.float32)
        
        saver = tf.train.Saver()
        
        loss = tf.losses.mean_squared_error(labels = ys, predictions = prediction)
        train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        #i = 0
        n,d = X.shape
        batch_size = 800
        for epoch in range(50): 
            for start in range(0,n,batch_size):
                end = start + batch_size
                sess.run(train_step, feed_dict={xs: x_data[start:end], ys: y_data[start:end]})
            
        print("LOSS:")
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        
        prediction_value = sess.run(prediction, feed_dict={xs: X_t})
        
        if(dataset == 2):
            save_path = saver.save(sess, './Allstate/model/allstate_NN'+str(k)+'state=100.ckpt') 
        
        sess.close()
        
    else:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X, y)
        
        
        if(dataset == 1):
            joblib.dump(reg, './Firstdata/firstdata_NN_'+str(k)+'class_state=100.pkl')
        if(dataset == 3):
            joblib.dump(reg, './FashionMNIST/fashionmnist_NN_'+str(k)+'class_state=100.pkl')
        if(dataset == 4):
            joblib.dump(reg, './KuzushijiMNIST/kuzushijimnist_NN_'+str(k)+'class_state=100.pkl')
        if(dataset == 5):
            joblib.dump(reg, './'+str(dataname)+'/'+str(dataname)+'_NN_'+str(k)+'class_state=100.pkl')
    
    
        prediction_value = reg.predict(X_t)
    
    return prediction_value

def nn_loss(X,y,X_t,k,dataset,dataname):
    
    if(dataset == 2):
        n,d = X.shape
        xs = tf.placeholder(tf.float32, [None, d])
        ys = tf.placeholder(tf.float32, [None, 4])
        
        l1 = tf.layers.dense(inputs = xs, units = n, activation = tf.nn.relu)
        l2 = tf.layers.dense(inputs = l1, units = 770, activation = tf.nn.relu)
        l3 = tf.layers.dense(inputs = l2, units = 386, activation = tf.nn.relu)
        prediction = tf.layers.dense(inputs = l3, units = 4, activation = None)
        
        x_data = X.astype(np.float32)
        y_data = y.astype(np.float32)
        
        saver = tf.train.Saver()
        
        loss = tf.losses.mean_squared_error(labels = ys, predictions = prediction)
        train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        #i = 0
        n,d = X.shape
        batch_size = 800
        for epoch in range(50): 
            for start in range(0,n,batch_size):
                end = start + batch_size
                sess.run(train_step, feed_dict={xs: x_data[start:end], ys: y_data[start:end]})
            
        print("LOSS:")
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        
        prediction_value = sess.run(prediction, feed_dict={xs: X_t})

        if(dataset == 2):
            save_path = saver.save(sess, './Allstate/model/allstate_loss'+str(k)+'state=100.ckpt')

        sess.close()
    
    else:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X, y)
        
        
        if(dataset == 1):
            joblib.dump(reg, './Firstdata/firstdata_LOSS_'+str(k)+'class_state=100.pkl')
        if(dataset == 3):
            joblib.dump(reg, './FashionMNIST/fashionmnist_LOSS_'+str(k)+'class_state=100.pkl')
        if(dataset == 4):
            joblib.dump(reg, './KuzushijiMNIST/kuzushijimnist_LOSS_'+str(k)+'class_state=100.pkl')
        if(dataset == 5):
            joblib.dump(reg, './'+str(dataname)+'/'+str(dataname)+'_LOSS_'+str(k)+'class_state=100.pkl')
    
    
        prediction_value = reg.predict(X_t)
    return prediction_value

def en(weight, y, ksd, gw, s):
    
    n, d = y.shape
    en_p = np.zeros((n,1))
    w = np.zeros((1,4))
    #en = np.zeros((n,1))
    #print(weight)
    
    #print(weight)
    for i in range(n):
        weight[i] =  weight[i] * ksd[i] * gw
        #w = np.power(5, weight[i])/sum(np.power(5, weight[i]))
        if(s == 1):
            w = np.exp(weight[i])/sum(np.exp(weight[i]))
        if(s == 0):
            w = weight[i]/sum(weight[i])
        
        en_p[i] = np.dot(w.T,y[i])
        
    return en_p

def mvb(knn,svc,dec,xgb):
    n, = knn.shape
    mvm = np.zeros((n,1))
    for i in range(n) :
        zcount = np.zeros((n,1))
        ocount = np.zeros((n,1))
        
        if (knn[i] == 0):
            zcount[i] += 1
        else:
            ocount[i] += 1
        if (svc[i] == 0):
            zcount[i] += 1
        else:
            ocount[i] += 1
        if (dec[i] == 0):
            zcount[i] += 1
        else:
            ocount[i] += 1
        if (xgb[i] == 0):
            zcount[i] += 1
        else:
            ocount[i] += 1

        if (zcount[i] > ocount[i]):
            mvm[i]=0
        else:
            mvm[i]=1
    
    return mvm

def savedata(X_train_c,X_train_n,y_train_c,y_train_n,X_test,y_test,knn_exac,svc_exac,dec_exac,xgb_exac,knn_yte,svc_yte,dec_yte,xgb_yte,knn_nKSD,svc_nKSD,dec_nKSD,xgb_nKSD,knn_tKSD,svc_tKSD,dec_tKSD,xgb_tKSD,knnp_n, knnp, knnpro_n, knnpro, svcp_n,svcp, svcpro_n, svcpro, decp_n, decp, decpro_n, decpro, xgbp_n, xgbp, xgbpro_n, xgbpro,trained_loss,trained_NN, trained_KSD,dataset,dataname,k):
    
    if(dataset == 1):
        values = X_train_c
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/X_train_c_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_train_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/X_train_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_c
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/y_train_c_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/y_train_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_test
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/X_test_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_test
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/y_test_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_predict_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_predict_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_predict_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_predict_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_predict_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_predict_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_predict_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_predict_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_predictpro_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_predictpro_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_predictpro_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_predictpro_n_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_predictpro_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_predictpro_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_predictpro_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_predictpro_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knn_exac
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_n_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_exac
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_n_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_exac
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_n_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_exac
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_n_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_yte
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_te_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_yte
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_te_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_yte
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_te_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_yte
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_te_exac_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_n_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_n_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_n_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_n_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/KNN_te_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/SVC_te_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/DEC_te_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/XGB_te_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_loss
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/Trained test_loss_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_NN
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/Trained test_NN_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = trained_KSD
        df = pd.DataFrame(values)
        df.to_csv("./Firstdata/Trained test_KSD_count_firstdata_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
    if(dataset == 2):
        values = X_train_c
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/X_train_c_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_train_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/X_train_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_c
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/y_train_c_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/y_train_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_test
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/X_test_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_test
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/y_test_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_predict_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_predict_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_predict_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_predict_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_predict_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_predict_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_predict_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_predict_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_predictpro_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_predictpro_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_predictpro_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro_n
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_predictpro_n_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_predictpro_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_predictpro_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_predictpro_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_predictpro_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knn_exac
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_n_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_exac
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_n_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_exac
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_n_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_exac
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_n_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_yte
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_te_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_yte
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_te_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_yte
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_te_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_yte
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_te_exac_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_n_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_n_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_n_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_n_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/KNN_te_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/SVC_te_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/DEC_te_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/XGB_te_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_loss
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/Trained test_loss_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_NN
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/Trained test_NN_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = trained_KSD
        df = pd.DataFrame(values)
        df.to_csv("./Allstate/Trained test_KSD_count_allstate_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
    if(dataset == 3):
        values = X_train_c
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/X_train_c_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_train_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/X_train_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_c
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/y_train_c_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/y_train_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_test
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/X_test_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_test
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/y_test_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_predict_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_predict_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_predict_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_predict_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_predict_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_predict_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_predict_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_predict_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_predictpro_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_predictpro_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_predictpro_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro_n
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_predictpro_n_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_predictpro_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_predictpro_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_predictpro_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_predictpro_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knn_exac
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_n_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_exac
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_n_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_exac
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_n_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_exac
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_n_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_yte
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_te_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_yte
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_te_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_yte
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_te_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_yte
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_te_exac_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_n_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_n_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_n_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_n_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/KNN_te_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/SVC_te_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/DEC_te_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/XGB_te_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_loss
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/Trained test_loss_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_NN
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/Trained test_NN_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = trained_KSD
        df = pd.DataFrame(values)
        df.to_csv("./FashionMNIST/Trained test_KSD_count_fashionmnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
    if(dataset == 4):
        values = X_train_c
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/X_train_c_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_train_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/X_train_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_c
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/y_train_c_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/y_train_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_test
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/X_test_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_test
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/y_test_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_predict_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_predict_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_predict_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_predict_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_predict_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_predict_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_predict_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_predict_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_predictpro_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_predictpro_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_predictpro_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro_n
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_predictpro_n_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_predictpro_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_predictpro_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_predictpro_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_predictpro_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knn_exac
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_n_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_exac
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_n_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_exac
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_n_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_exac
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_n_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_yte
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_te_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_yte
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_te_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_yte
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_te_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_yte
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_te_exac_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_n_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_n_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_n_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_n_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/KNN_te_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/SVC_te_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/DEC_te_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/XGB_te_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_loss
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/Trained test_loss_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_NN
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/Trained test_NN_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = trained_KSD
        df = pd.DataFrame(values)
        df.to_csv("./KuzushijiMNIST/Trained test_KSD_count_kuzushijimnist_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
    if(dataset == 5):
        values = X_train_c
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/X_train_c_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_train_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/X_train_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_c
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/y_train_c_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_train_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/y_train_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = X_test
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/X_test_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = y_test
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/y_test_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_predict_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_predict_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_predict_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_predict_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knnp
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_predict_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decp
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_predict_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcp
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_predict_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbp
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_predict_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_predictpro_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_predictpro_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_predictpro_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro_n
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_predictpro_n_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knnpro
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_predictpro_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = decpro
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_predictpro_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svcpro
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_predictpro_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgbpro
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_predictpro_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = knn_exac
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_n_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_exac
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_n_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_exac
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_n_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_exac
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_n_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_yte
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_te_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_yte
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_te_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_yte
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_te_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_yte
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_te_exac_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_n_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_n_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_n_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_nKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_n_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = knn_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/KNN_te_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = svc_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/SVC_te_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = dec_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/DEC_te_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = xgb_tKSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/XGB_te_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_loss
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/Trained test_loss_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
        values = trained_NN
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/Trained test_NN_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
    
        values = trained_KSD
        df = pd.DataFrame(values)
        df.to_csv("./"+str(dataname)+"/Trained test_KSD_count_"+str(dataname)+"_"+ str(k) +"class_state=100.csv", sep=',',index = None)
        
    return 0

def main():
    ce = 0
    de = 0
    data = pd.read_csv('setup.txt', header = None, delimiter = " ")
    data = data.values
    
    mode = data[0,2]                #set numbers of class (>=2)
    
    if(int(mode) <2):
        print("Wrong class numbers")
        ce = 1
    
    dataset = int(data[1,2])        #set data set
    if(5 < dataset):
        print("Wrong dataset")
        de = 1
    if(dataset < 1):
        print("Wrong dataset")
        de = 1
        
    dataname = data[2,2]            #your data name
    
    if(ce == 0 and de == 0):
        mn_train, mn_test = readdata(dataset,dataname)
        y_test = mn_test[:,0].reshape(-1,1)
        print("DATA OK")
        knnp_n, knnp, knnpro_n, knnpro, svcp_n,svcp, svcpro_n, svcpro, decp_n, decp, decpro_n, decpro, xgbp_n, xgbp, xgbpro_n, xgbpro, trained_loss, trained_NN, trained_KSD = buildenmen(mn_train, mn_test, mode,dataset,dataname)
    
        allmodel_en(y_test,mode, dataset,dataname)
    
    return 0

main()