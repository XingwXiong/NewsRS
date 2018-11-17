# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:03:38 2018

@author: 温旭
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn import preprocessing
from sklearn.decomposition import NMF

path=''
train_data = pd.read_csv(path+'train.csv',encoding='utf-8')
test_data = pd.read_csv(path+'train.csv',encoding='utf-8')
n_users = train_data.usr_id.unique().shape[0]
n_items = train_data.news_id.unique().shape[0]
print(n_users)
print(n_items)
le_usr = preprocessing.LabelEncoder()
le_news = preprocessing.LabelEncoder()
le_usr.fit(train_data["usr_id"])
le_news.fit(train_data["news_id"])
train_data["usr_id"] = le_usr.transform(train_data["usr_id"])
train_data["news_id"] = le_news.transform(train_data["news_id"])

#创建user-item矩阵
train_data_matrix = np.zeros((n_users, n_items))
for row in data.itertuples(index=True, name='Pandas'):
    train_data_matrix[row[1], row[2]] += 1

print(train_data_matrix.sum())
print(train_data_matrix.max())

#SVD
U,S,VT = svds(train_data_matrix, k=20)#用户主题分布，奇异值，物品主题分布
S= np.diag(S)
reconstruct_matrix=np.dot(np.dot(U, S), VT) 
filter_matrix = train_data_matrix < 1e-6 
svd_result=reconstruct_matrix* filter_matrix
print(svd_result)

#NMF
nmf = NMF(n_components=2)
user_distribution = nmf.fit_transform(train_data_matrix)
item_distribution = nmf.components_
reconstruct_matrix = np.dot(user_distribution, item_distribution)
filter_matrix = train_data_matrix < 1e-6  
nmf_result=reconstruct_matrix*filter_matrix
print(nmf_result)


train_data["usr_id"] = le_usr.inverse_transform(train_data["usr_id"])
train_data["news_id"] = le_news.inverse_transform(train_data["news_id"])
