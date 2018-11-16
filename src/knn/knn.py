'''
    Author:         Xingw Xiong
    Data:           2018/11/16
    Description:    KNN
'''

import pandas as pd
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ROOT_DIR = '../../'

news_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'news_vector.csv'))
usr_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'usr_vector.csv'))
usr_mean = usr_vecs.mean(1)

train = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'train_jieba.csv'))
test  = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'test_jieba.csv'))


'''
    Di Function:    Square Distance
    Description:    Return top k nearest items
'''
def knn(vec, k):
    dif_vecs = news_vecs.sub(vec, axis=0)
    dif_vecs = dif_vecs**2
    d_dis = {}
    for cname in news_vecs.columns.values:
        d_dis[cname] = dif_vecs[cname].sum()
    d_dis = sorted(d_dis.items(), key=lambda d: d[1])
    return d_dis[:k]

def recommend(usr_id):
    vec = usr_vecs[usr_id] if usr_id in usr_vecs.columns.values else usr_mean 
    return knn(vec, k=10)

#tot_cnt = test.shape[0]
tot_cnt = 10
hit_cnt = 0

for i in range(tot_cnt):
    news_dis = recommend(test.iloc[i]['usr_id'])
    news_set = [x[0] for x in news_dis]
    logging.info(news_set)
    logging.info(test.iloc[i]['news_id'])
    if test.iloc[i]['news_id'] in news_set:
        hit_cnt = hit_cnt + 1

logging.info("[tot: %d, hit: %d]" % (tot_cnt, hit_cnt))
    #logging.info(test.iloc[0])
    #logging.info(news_set)
    #logging.info(news_dis)
