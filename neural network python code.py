# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 23:16:24 2022

@author: abdal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
adult_data=pd.read_csv(r'...\adultdata.csv',header=None)
adult_train_x=adult_data[adult_data.columns[0:14]]
adult_train_y=adult_data[adult_data.columns[14:15]]
adult_train_x_encoded=pd.get_dummies(adult_train_x,columns=[1,3,5,6,7,8,9,13])
adult_train_x_encoded0=adult_train_x_encoded
adult_train_y_encoded=pd.get_dummies(adult_train_y,columns=[14])
adult_train_x_encoded=StandardScaler().fit_transform(adult_train_x_encoded)
mlp_classifer=MLPClassifier(hidden_layer_sizes=(54,54),max_iter=2000)
mlp_classifer.fit(adult_train_x_encoded,adult_train_y_encoded)
mlp_classifer.score(adult_train_x_encoded,adult_train_y_encoded)
###testing decision tree accuracy
adult_data_test=pd.read_csv(r'...adulttest.csv',header=None)
adult_train_test_x=adult_data_test[adult_data_test.columns[0:14]]
adult_train_test_y=adult_data_test[adult_data_test.columns[14:15]]
adult_train_x_test_encoded=pd.get_dummies(adult_train_test_x,columns=[1,3,5,6,7,8,9,13])
adult_train_y_test_encoded=pd.get_dummies(adult_train_test_y,columns=[14])
adult_train_x_test_encoded=adult_train_x_test_encoded.reindex(columns=adult_train_x_encoded0.columns, fill_value=0)
adult_train_x_test_encoded=StandardScaler().fit_transform(adult_train_x_test_encoded)
mlp_classifer.score(adult_train_x_test_encoded,adult_train_y_test_encoded)
