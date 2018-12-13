#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys, logging, math, time
import matplotlib.pyplot as plt

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_dir='../../'

data=pd.read_csv('%s/data/%s'%(root_dir,'user_click_data.txt'),encoding='utf-8',delimiter="\t",
        names=["usr_id","news_id","scan_time","news_title","content","publish_time"],)
#        usecols=[0,1,2,])
#data=data.sample(n=10000, replace=False)
news=data[['news_id', 'news_title', 'content', 'publish_time']].drop_duplicates('news_id')
logging.info(['data shape', data.shape])
logging.info(['news shape', news.shape])

# Data Clean
data=data[data['publish_time']!='NULL']
data.dropna(axis=0, how='any', inplace=True)
del_list=[]
for ix, x in zip(range(data.shape[0]), data['publish_time']):
    try:
        time.mktime(time.strptime(x, '%Y年%m月%d日%H:%M'))
    except:
        del_list.append(ix)

logging.info("del: %d" % len(del_list))
data.drop(data.index[del_list], inplace=True)

data['publish_time'] = data['publish_time'].apply(lambda x : int(time.mktime(time.strptime(x, '%Y年%m月%d日%H:%M'))))

data.sort_values(by='publish_time', inplace=True)
logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['publish_time'].iloc[int(data.shape[0] * 0.05)])))
logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['publish_time'].min())))
logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['publish_time'].max())))

logging.info(['data shape', data.shape])
news=data[['news_id', 'news_title', 'content', 'publish_time']].drop_duplicates('news_id')
logging.info(['news shape', news.shape])
news=news[news['publish_time']>=data['publish_time'].iloc[int(data.shape[0] * 0.05)]]
logging.info(['news shape', news.shape])


#根据时间划分训练集和测试集
min_time=data['scan_time'].min(); max_time=data['scan_time'].max()
logging.info((max_time-min_time)/3600.0/24)
divide_time=min_time+21*3600*24
train=data[data["scan_time"]<divide_time]
test=data[data["scan_time"]>=divide_time]

