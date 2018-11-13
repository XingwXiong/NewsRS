# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:34:08 2018

@author: 温旭
"""
import jieba
import pandas as pd
import numpy as np
import re
import sys
import gensim
from sklearn.decomposition import NMF

path=''
#用户编号,新闻编号,浏览时间,新闻标题,新闻详细内容,新闻发表时间
data = pd.read_csv(path+'user_click_data.txt',encoding='utf-8',delimiter="\t",
                   names=["usr_id", "news_id", "scan_time","news_title"
                          ,"content","publish_time"])

"""
print(data.head(1))
print("用户情况：")
print(data['usr_id'].value_counts())
print("新闻情况：")
print(data['news_id'].value_counts())
"""
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

#print(train_original)

#删掉404的行（标题404的内容一定NULL,内容或时间出现NULL的可以用）
train_data=train_original[~train_original["news_title"].str.contains('404')]
test_data=test_original[~test_original["news_title"].str.contains('404')]

data_404=data[data["news_title"].str.contains('404')]
#print(data_404)
train_original.to_csv("train_original.csv", index=False)
test_original.iloc[:, 0:2].to_csv("test_original.csv", index=False)

#删掉字符,只保留中文英文和数字
def delete_char(line): 
    rule=re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
    line=re.sub(rule,"",line)
    return line

#分词 直接用结巴分词
def cutword(line):
    tmp=jieba.cut(line)
    return ' '.join(tmp)

print("start deleting")
train_data["content"] = train_data["content"].astype(str)
train_data['news_title']=train_data['news_title'].astype(str)
train_data["content"] = train_data["content"].apply(delete_char)
train_data['news_title']=train_data['news_title'].apply(delete_char)
print("start cutting")
#t=pd.DataFrame(train_data['content'].astype(str))
train_data['word_title']=train_data['news_title'].apply(cutword)
train_data['word_content']=train_data['content'].apply(cutword)

print("start outputting")
train_data.iloc[:, [0,1,2,6,7,5]].to_csv("train.csv", index=False)
test_data.iloc[:, 0:2].to_csv("test.csv", index=False)


    
