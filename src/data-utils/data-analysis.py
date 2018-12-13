#!/usr/bin/env python3
'''
    Author:         XingwXiong
    Description:    Feature Analysis
'''
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys, logging, math
import matplotlib.pyplot as plt
from gensim.models import Doc2Vec

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_dir='../../'

data=pd.read_csv('%s/data/%s'%(root_dir,'user_click_data.txt'),encoding='utf-8',delimiter="\t",
        names=["usr_id","news_id","scan_time","news_title","content","publish_time"],)
#        usecols=[0,1,2,])

#data=data.sample(n=10000, replace=False)

def scan_time_analysis():
    def split_by_days(data, t):
        min_time=data['scan_time'].min()
        max_time=data['scan_time'].max()
        circle = math.ceil((max_time - min_time)/3600.0/24/t)
        data_by_days = []
        for i in range(circle):
            data_by_days.append(data[data['scan_time'] < min_time + (i+1)*3600*24*t])
        return data_by_days

    data['usr_id']=data['usr_id'].apply(lambda x : usr_mp[x])
    data['news_id']=data['news_id'].apply(lambda x : news_mp[x])

    usr_id=data['usr_id'].copy().sort_values().unique()
    news_id=data['news_id'].copy().sort_values().unique()

    usr_mp= { usr_id[i]:i for i in range(len(usr_id)) }
    news_mp={ news_id[i]:i for i in range(len(news_id)) }

    #根据时间划分训练集和测试集
    min_time=data['scan_time'].min(); max_time=data['scan_time'].max()
    logging.info((max_time-min_time)/3600.0/24)
    divide_time=min_time+21*3600*24
    train=data[data["scan_time"]<divide_time]
    test=data[data["scan_time"]>=divide_time]

    #plt.scatter(data['usr_id'], data['news_id'])

    train_by_days = split_by_days(data, 10)

    plt.figure()
    plt.scatter(train['usr_id'], train['news_id'])

    for d in train_by_days:
        plt.figure(figsize=(3,3))
        plt.scatter(d['usr_id'], d['news_id'])

    #plt.figure()
    #plt.scatter(test['usr_id'], test['news_id'])

    #doc_model=Doc2Vec.load('%s/model/news_doc.model'%root_dir)

    plt.show()

