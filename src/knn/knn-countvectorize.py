#!/usr/bin/env python3
'''
    Author:         Xingw Xiong
    Data:           2018/11/16
    Description:    KNN-Scipy
'''
import numpy as np
import pandas as pd
import time, logging, re
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_dir = '../../'
K = 10

# Reading Data
data = pd.read_csv('%s/data/%s' % (root_dir, 'clean_user_click_data.csv'), encoding='utf-8')
news = data[['news_id', 'news_title', 'content', 'publish_time', 'jieba_title', 'jieba_content']].copy().drop_duplicates('news_id')
news_title_mp = {idx:title for idx,title in zip(news['news_id'], news['news_title'])}

# Remove the data without correct publish time
del_list=[]
for ix, x in zip(range(data.shape[0]), data['publish_time']):
    try:
        time.mktime(time.strptime(x, '%Y年%m月%d日%H:%M'))
    except:
        del_list.append(ix)
data.drop(data.index[del_list], inplace=True)

news = data[['news_id', 'news_title', 'content', 'publish_time', 'jieba_title', 'jieba_content']].copy().drop_duplicates('news_id')
usrs = data['usr_id'].copy().unique()

# loading stopwords
infile = open('%s/data/stopwords.txt' % root_dir, encoding='utf-8')
stopwords = [x.replace('\n', ' ').strip() for x in infile.readlines()]

# Get News Vector By Corpus
vectorizer = CountVectorizer(stop_words=stopwords, max_features=1000)
#news_vecs = vectorizer.fit_transform(news['jieba_title'].tolist()).todense()
news_vecs = vectorizer.fit_transform((news['jieba_title']+news['jieba_content']).astype(str).tolist()).todense()
news_vecs_mp = {idx:vec for idx,vec in zip(news['news_id'], news_vecs)}

# Get User Vector By News Vector
user_vecs_mp = {}
grouped = data.groupby('usr_id')
for usr_id, group in grouped:
    user_vecs_mp[usr_id] = group['news_id'].apply(lambda x : news_vecs_mp[x]).mean()

# Get news after 2014-1-1
news['publish_time'] = news['publish_time'].apply(lambda x : int(time.mktime(time.strptime(x, '%Y年%m月%d日%H:%M'))))
news_filter = news['publish_time'] >= int(time.mktime(time.strptime('2014', '%Y'))) 
news = news[news_filter]
news_vecs = news_vecs[news_filter]

# Build KNN
nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(X=news_vecs, y=news['news_id'].tolist())
#nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(X=news_vecs)

# Get the hotest K items
train = pd.read_csv('%s/data/clean_train_data.csv' % root_dir)
hotest_k = train['news_id'].value_counts().index[0:K]

def recommend(usr_id, k=K):
    if usr_id in user_vecs_mp.keys():
        idx, =  nbrs.kneighbors(X=user_vecs_mp[usr_id].tolist(), n_neighbors=k, return_distance=False)
        return list(map(int, [news.iloc[i]['news_id'] for i in idx]))
    else:
        return hotest_k 

logging.info('===============================')
hit_cnt = 0
test = pd.read_csv('%s/data/clean_test_data.csv' % root_dir)
test = test.sample(n=10, replace=False)
for i in range(test.shape[0]):
    query = test.iloc[i]
    r_set = recommend(test.iloc[i]['usr_id'])
    # print(r_set, type(r_set))
    if query['news_id'] in r_set:
        hit_cnt = hit_cnt + 1
        print("HHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTT: %d/%d" % (hit_cnt, i+1))
        logging.info('#{}:'.format(i+1))
        logging.info("[usr_id: {}, news_id:{}, new_title:{}]".format(query['usr_id'], query['news_id'], news_title_mp[query['news_id']]))
        for r in r_set:
            logging.info("[news_id:{}, new_title:{}]".format(r, news_title_mp[r]))
logging.info("[tot: %d, hit: %d]" % (test.shape[0], hit_cnt))

