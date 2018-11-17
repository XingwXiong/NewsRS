# NewsRS: News Recommendation System

## STARTUP

### Initialization
```
pip3 install jieba,numpy,pandas,gensim  # or conda install ...
```

```
$ git init
$ git remote add origin git@github.com:XingwXiong/NewsRS.git
$ git pull origin master
$ cd data; tar xzf user_click_data.tar.gz
```


### Classification
- `data/news_jieba.csv` : pandas.DataFrame 格式, 对数据集中的所有新闻 `jieba` 分词; 
- `data/news_vector.csv`: pandas.DataFrame 格式, 列名为`news_id`, 每一列是长度为`100`的news向量(news 向量是通过`gensim.model.Doc2Vec`得到的);
- `data/usr_vector.csv`:  pandas.DataFrame 格式, 列名为`usr_id`, 每一列是长度为`100`的usr向量(usr 向量是由训练数据中每个用户访问的所有新闻的向量平均值得来的);
