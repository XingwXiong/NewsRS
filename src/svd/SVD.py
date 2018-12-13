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
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

a=datetime.now()

path=''
train_data = pd.read_csv(path+'train.csv',encoding='utf-8')
test_data = pd.read_csv(path+'test.csv',encoding='utf-8')

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
#注意此矩阵行列从0开始，对id进行label encoder
train_data_matrix = np.zeros((n_users, n_items))
for row in train_data.itertuples(index=True, name='Pandas'):
    train_data_matrix[row[1], row[2]] += 1

print(train_data_matrix.sum())
print(train_data_matrix.max())


#SVD
n_components=30
U,S,VT = svds(train_data_matrix, k=n_components)#用户主题分布，奇异值，物品主题分布
S= np.diag(S)
reconstruct_matrix=np.dot(np.dot(U, S), VT) 
filter_matrix = train_data_matrix < 1e-6 
svd_result=reconstruct_matrix* filter_matrix
print(svd_result)
#print(svd_result.max())
'''
#SVD feature

#NMF
n_components=30#主题数目
nmf = NMF(n_components)
user_distribution = nmf.fit_transform(train_data_matrix)
item_distribution = nmf.components_
reconstruct_matrix = np.dot(user_distribution, item_distribution)
filter_matrix = train_data_matrix < 1e-6  
nmf_result=reconstruct_matrix*filter_matrix
print(nmf_result)
#print(nmf_result.max())
'''
train_data["usr_id"] = le_usr.inverse_transform(train_data["usr_id"])
train_data["news_id"] = le_news.inverse_transform(train_data["news_id"])


def recommendNews(K,result_matrix):
    #推荐K个新闻
    [m,n]=result_matrix.shape
    recommend=np.zeros((m,K))
    #最热的K个新闻
    train_data_recent=train_data[train_data["publish_time"]>"2014年03月01日"]
    hot_k=train_data_recent["news_id"].value_counts().index[0:K]
    
    for i in range(m):
        arr=result_matrix[i]
        res=arr.argsort()[-K:][::-1]
        recommend[i]=res  
    
    recommend=recommend.astype(int)
    for i in range(m):
        recommend[i]=le_news.inverse_transform(recommend[i])
    print(recommend)
    res={}
    for usr in test_data.usr_id.unique():
        #res[usr]=hot_k
        
        if(usr in train_data.usr_id.unique()):       
            tmp=le_usr.transform([usr])#usr_id变为矩阵行号0
            predict=recommend[tmp]
            res[usr]=predict
        else:
            res[usr]=hot_k
            
    #print(res)
    
    cnt=0
    for i in range(test_data.shape[0]):
        if test_data.iloc[i]['news_id'] in res[test_data.iloc[i]['usr_id']]:
            cnt=cnt+1
    print(cnt)
#recommendNews(10,nmf_result)
recommendNews(10,svd_result)

b=datetime.now()
print((b-a).seconds)