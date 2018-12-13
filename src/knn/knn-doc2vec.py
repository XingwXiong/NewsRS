#!/usr/bin/env python3
'''
    Author:         Xingw Xiong
    Data:           2018/11/16
    Description:    KNN-Scipy
'''
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time, logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ROOT_DIR = '../../'
K = 40

#news_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'news_vector.csv'))
#usr_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'usr_vector.csv'))
news_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'news_vector_title.csv'))
usr_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'usr_vector_title.csv'))
usr_mean = usr_vecs.mean(1)

news  = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'news_jieba.csv'), names=['news_id','news_title','content','publish_time'])
# Data Clean
news.dropna(axis=0, how='any', inplace=True)
del_list=[]
for ix, x in zip(range(news.shape[0]), news['publish_time']):
    try:
        time.mktime(time.strptime(x, '%Y年%m月%d日%H:%M'))
    except:
        del_list.append(ix)

logging.info("del: %d" % len(del_list))
news.drop(news.index[del_list], inplace=True)

news['publish_time'] = news['publish_time'].apply(lambda x : int(time.mktime(time.strptime(x, '%Y年%m月%d日%H:%M'))))
news=news[news['publish_time'] >= int(time.mktime(time.strptime('2014', '%Y')))]

del_cols=[]
for ix, col in news_vecs.iteritems():
    if not ix in news['news_id'].tolist():
        del_cols.append(ix)
news_vecs.drop(del_cols, axis=1, inplace=True)

X=[col.tolist() for ix, col in news_vecs.iteritems()]
nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(X)
#print(X)

logging.info("KNN samples: %d" % len(X))

'''
    Di Function:    Square Distance
    Description:    Return top k nearest items
'''

def recommend(usr_id, k=K):
    vec = usr_vecs[usr_id] if usr_id in usr_vecs.columns.values else usr_mean 
    return nbrs.kneighbors(X=[vec.tolist(),], n_neighbors=k, return_distance=False)

hit_cnt = 0

#train = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'train_jieba.csv'))
test  = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'test_jieba.csv'))
test = test.sample(n=20, replace=False)

news_mp = { int(e['news_id']):e['news_title'] for ix, e in news.iterrows()}

print(news_mp)
for i in range(test.shape[0]):
    idx, = recommend(test.iloc[i]['usr_id'])
    news_set = list(map(int,[news_vecs.columns[i] for i in idx]))
    logging.info([x for x in news_set])
    logging.info([news_mp[x] for x in news_set])
    logging.info(news[test.iloc[i]['news_id']])
    if test.iloc[i]['news_id'] in news_set:
        hit_cnt = hit_cnt + 1
        print("HHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTT: %d" % hit_cnt)

logging.info("[tot: %d, hit: %d]" % (test.shape[0], hit_cnt))
    #logging.info(test.iloc[0])
    #logging.info(news_set)
    #logging.info(news_dis)
