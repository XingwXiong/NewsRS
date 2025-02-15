#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Nov 13 18:34:08 2018
Author: 温旭, Xingwang Xiong

'''

import pandas as pd
import numpy as np
import re, sys, os, multiprocessing
import jieba
import gensim
from gensim.models import Word2Vec, Doc2Vec
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


root_dir='../../'
#用户编号,新闻编号,浏览时间,新闻标题,新闻详细内容,新闻发表时间
data = pd.read_csv('%s/data/%s'%(root_dir,'user_click_data.txt'),encoding='utf-8',delimiter="\t",
                   names=["usr_id", "news_id", "scan_time","news_title",
                          "content","publish_time"])

"""
print(data.head(1))
print("用户情况：")
print(data['usr_id'].value_counts())
print("新闻情况：")
print(data['news_id'].value_counts())
"""

'''
    将数据分成训练集和测试集
'''
if not os.path.exists('%s/data/train_original.csv'%root_dir) or \
    not os.path.exists('%s/data/test_original.csv'%root_dir) or \
    not os.path.exists('%s/data/news_original.csv'%root_dir) or \
    not os.path.exists('%s/data/404_original.csv'%root_dir):
    max_time=data["scan_time"].max()
    #print(max_time)
    min_time=data["scan_time"].min()
    #print(min_time)
    #print((max_time-min_time)/(3600*24))
    #根据时间划分训练集和测试集
    divide_time=min_time+21*3600*24
    #print(divide_time)#1395417620
    train_original=data[data["scan_time"]<divide_time]
    test_original=data[data["scan_time"]>=divide_time]
    _404_original=data[data["news_title"].str.contains('404')]
    news_original=data.drop(["usr_id", "scan_time"], axis=1).drop_duplicates("news_id")
    logging.info("[news number:%d]" % news_original.shape[0])

    train_original.to_csv('%s/data/train_original.csv'%root_dir, index=False)
    test_original.to_csv('%s/data/test_original.csv'%root_dir, index=False)
    _404_original.to_csv('%s/data/404_original.csv'%root_dir, index=False)
    news_original.to_csv('%s/data/news_original.csv'%root_dir, index=False)

'''
数据清洗+jieba分词
处理结果存在:ROOT_DIR/data/*_jieba.csv
'''

#删掉字符,只保留中文英文和数字
def delete_char(line): 
    return re.sub(re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]"),"",line)

#分词 直接用结巴分词
def cutword(line):
    return ' '.join(jieba.cut(line))

if not os.path.exists('%s/data/%s.csv'%(root_dir, 'train_jieba')) or \
    not os.path.exists('%s/data/%s.csv'%(root_dir, 'test_jieba')) or \
    not os.path.exists('%s/data/%s.csv'%(root_dir, 'news_jieba')):
    logging.info('start loading original data')
    train_data=pd.read_csv('%s/data/train_original.csv'%root_dir)
    test_data=pd.read_csv('%s/data/test_original.csv'%root_dir)
    news_data=pd.read_csv('%s/data/news_original.csv'%root_dir)

    #删掉404的行（标题404的内容一定NULL,内容或时间出现NULL的可以用）
    train_data=train_data[~train_data["news_title"].str.contains('404')]
    test_data=test_data[~test_data["news_title"].str.contains('404')]
    data=data[~data["news_title"].str.contains('404')]

    logging.info("start deleting")
    news_data["content"]=news_data["content"].astype(str)\
            .apply(lambda line : re.sub(re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]"),"",line))
    news_data["news_title"]=news_data["news_title"].astype(str).apply(delete_char)

    train_data["content"] = train_data["content"].astype(str).apply(delete_char)
    train_data['news_title']=train_data['news_title'].astype(str).apply(delete_char)

    logging.info("start cutting")
    news_data['news_title']=news_data['news_title'].apply(cutword)
    news_data['content']=news_data['content'].apply(cutword)
    news_data.rename(columns={'news_title':'word_title', 'content': 'word_content'}, inplace = True)
    #t=pd.DataFrame(train_data['content'].astype(str))
    train_data['news_title']=train_data['news_title'].apply(cutword)
    train_data['content']=train_data['content'].apply(cutword)
    train_data.rename(columns={'news_title':'word_title', 'content': 'word_content'}, inplace = True)

    logging.info("start outputting jieba")
    train_data.to_csv('%s/data/train_jieba.csv'%root_dir, index=False)
    test_data.iloc[:, 0:2].to_csv('%s/data/test_jieba.csv'%root_dir, index=False)
    news_data.to_csv('%s/data/news_jieba.csv'%root_dir, index=False)


'''
    Word2Vec    将分词结果转为词向量
    Doc2Vec     将分词结果转为句向量
'''

if not os.path.exists('%s/model/news_word_title.model'%root_dir) or \
    not os.path.exists('%s/model/news_doc_title.model'%root_dir):

    logging.info('start loading news_jieba.csv')
    #train_data=pd.read_csv('%s/data/%s.csv'%(root_dir, 'train_jieba'))
    news_data=pd.read_csv('%s/data/%s.csv'%(root_dir, 'news_jieba'))
    
    class NewsWordSentences(object):
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            for line in self.data:
                yield line.split()

    logging.info("start word2vec")
    news_text=(news_data['word_title']).astype(str)
    #news_text=(news_data['word_title']+news_data['word_content']).astype(str)
    word_sentences=NewsWordSentences(news_text)
    word_model=Word2Vec(word_sentences)
    word_model.save('%s/model/news_word_title.model'%root_dir)

    class NewsDocSentences(object):
        def __init__(self, data):
            self.data = data.head()

        def __iter__(self):
            for index,line in self.data.iterrows():
                yield gensim.models.doc2vec.LabeledSentence(words=line['news_text'].split(),
                        tags=[line['news_id']])

    logging.info('start doc2vec')
    doc_sentences=NewsDocSentences(pd.DataFrame({'news_id':news_data['news_id'], 'news_text':news_text}))
    #doc_model=Doc2Vec(doc_sentences, vector_size = 50, window = 5, min_count = 5, max_vocab_size = 10000, workers=multiprocessing.cpu_count())
    doc_model=Doc2Vec(doc_sentences, vector_size = 300, window = 5, min_count = 1, max_vocab_size = 50000, workers=multiprocessing.cpu_count())
    doc_model.save('%s/model/news_doc_title.model'%root_dir)
    # If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
    doc_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

if not os.path.exists('%s/data/news_vector_title.csv'%root_dir):
    logging.info('start loading news_jieba.csv')
    news_data=pd.read_csv('%s/data/%s.csv'%(root_dir, 'news_jieba'))
    #word_model=Word2Vec.load('%s/model/news_word.model'%root_dir)
    doc_model=Doc2Vec.load('%s/model/news_doc_title.model'%root_dir)
    logging.info('news_doc_title.model was loaded')

    logging.info('start building news vector')
    d_news_vec = {}
    for i, x in news_data.iterrows():
        text = str(x['word_title']) 
        #text = str(x['word_title']) + str(x['word_content']) 
        d_news_vec[x['news_id']] = doc_model.infer_vector(text.split())
    news_vecs = pd.DataFrame.from_dict(d_news_vec)
    news_vecs.to_csv('%s/data/news_vector_title.csv'%root_dir, index=False)

# Desciption: Get user vector
if not os.path.exists('%s/data/usr_vector_title.csv'%root_dir):
    logging.info('start loading news_vector_title.csv')
    news_vecs = pd.read_csv('%s/data/news_vector_title.csv'%root_dir)
    logging.info('news_vector_title.csv was loaded! ' + str(news_vecs.shape))

    logging.info('start loading train_jieba.csv')
    train_data=pd.read_csv('%s/data/%s.csv'%(root_dir, 'train_jieba'))
    d_usr_vec = {}
    d_usr_cnt = {}
    for i, x in train_data.iterrows():
        news_id = str(x['news_id'])
        usr_id = str(x['usr_id'])
        if not usr_id.isdigit():
            logging.warning(x)
        if usr_id not in d_usr_vec.keys():
            d_usr_vec[usr_id] = news_vecs[news_id]
            d_usr_cnt[usr_id] = 1
        else:
            d_usr_vec[usr_id] = (news_vecs[news_id] + \
                    d_usr_vec[usr_id] * d_usr_cnt[usr_id]) / (d_usr_cnt[usr_id] + 1)
            d_usr_cnt[usr_id] = d_usr_cnt[usr_id] + 1

    usr_vecs = pd.DataFrame.from_dict(d_usr_vec)
    logging.info('usr_vector_title.csv outputting ' + str(usr_vecs.shape))
    usr_vecs.to_csv('%s/data/%s'%(root_dir, 'usr_vector_title.csv'), index=False)

