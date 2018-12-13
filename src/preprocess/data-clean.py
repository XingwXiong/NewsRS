#!/usr/bin/env python3
'''
    Author:         Xingw Xiong
    Data:           2018/11/16
    Description:    Data Clean 
'''
import numpy as np
import pandas as pd
import logging, re
import jieba 

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_dir = '../../'

# Reading Data
data = pd.read_csv('%s/data/%s' % (root_dir, 'user_click_data.txt'), encoding='utf-8', delimiter='\t', names=['usr_id', 'news_id', 'scan_time', 'news_title','content','publish_time'])

# Data Clean
logging.info('start data clean, data size:{}'.format(data.shape))
data.dropna(axis=0, how='any', inplace=True)
data=data[~data['news_title'].str.contains('404')]
data['news_title']=data['news_title'].astype(str).apply(lambda x:re.sub(re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]"),'', x))
data['content']=data['content'].astype(str).apply(lambda x:re.sub(re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]"),'', x))
logging.info('finish data clean, data size:{}'.format(data.shape))

# jieba Chinese Word Segmentation
logging.info('start cutting')
data['jieba_title']=data['news_title'].apply(lambda x : ' '.join(jieba.cut(x)))
data['jieba_content']=data['content'].apply(lambda x : ' '.join(jieba.cut(x)))
logging.info('finish cutting')

# split data
max_time = data['scan_time'].max()
min_time = data['scan_time'].min()
div_time = min_time + 21 * (3600*24)
train = data[data['scan_time']< div_time]
test  = data[data['scan_time']>=div_time]

# save result after data cleaning
logging.info('start saving')
data.to_csv('%s/data/clean_user_click_data.csv' % root_dir, index=False)
train.to_csv('%s/data/clean_train_data.csv' % root_dir, index=False)
test.to_csv('%s/data/clean_test_data.csv' % root_dir, index=False)
logging.info('finish saving')
