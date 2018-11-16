'''
    Author:         Xingw Xiong
    Data:           2018/11/16
    Description:    KNN
'''

import pandas as pd

ROOT_DIR = '../../data'

news_vecs = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'news_vector.csv'))

train = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'train_jieba.csv'))
test  = pd.read_csv('%s/data/%s' % (ROOT_DIR, 'test_jieba.csv'))


'''
    Description:    Return top k nearest items
'''
def knn(vec, k):
    pass

def recommend(usr_id):
    
