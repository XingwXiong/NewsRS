#!/usr/bin/env python3
'''
    Author:         Xingw Xiong
    Data:           2018/11/16
    Description:    KNN-Scipy
'''
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ROOT_DIR = '../../'
K = 40

news_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'news_vector.csv'))
usr_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'usr_vector.csv'))
#news_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'news_vector_title.csv'))
#usr_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'usr_vector_title.csv'))
usr_mean = usr_vecs.mean(1)

train = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'train_jieba.csv'))
test  = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'test_jieba.csv'))

#logging.info([col.tolist() for ix, col in news_vecs.T.head(1).T.iteritems()])


news_id = list(map(int, news_vecs.columns))
news_id = np.sort(news_id)
news_id = news_id[-int(len(news_id)*0.5):]
X=[]
for idx in news_id:
    X.append(news_vecs[str(idx)].tolist())

#nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit([col.tolist() for ix, col in news_vecs.iteritems()])
nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(X)

logging.info("KNN samples: %d" % len(X))

'''
    Di Function:    Square Distance
    Description:    Return top k nearest items
'''

def recommend(usr_id, k=K):
    vec = usr_vecs[usr_id] if usr_id in usr_vecs.columns.values else usr_mean 
    return nbrs.kneighbors(X=[vec.tolist(),], n_neighbors=k, return_distance=False)

hit_cnt = 0

#test = test.sample(n=1000, replace=False)

for i in range(test.shape[0]):
    idx, = recommend(test.iloc[i]['usr_id'])
    news_set = list(map(int,[news_vecs.columns[i] for i in idx]))
    #logging.info(news_set)
    #logging.info(test.iloc[i]['news_id'])
    if test.iloc[i]['news_id'] in news_set:
        hit_cnt = hit_cnt + 1
        #print("HHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTT")

logging.info("[tot: %d, hit: %d]" % (test.shape[0], hit_cnt))
    #logging.info(test.iloc[0])
    #logging.info(news_set)
    #logging.info(news_dis)
