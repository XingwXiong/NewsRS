# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF
import numpy as np
from sklearn import preprocessing
from gensim import corpora,similarities,models
# import jieba

root_dir = '../../'

def to_matrix(df):
    n_users = df.usr_id.unique().shape[0]
    n_news = df.news_id.unique().shape[0]
    matrix = np.zeros((n_users, n_news))
    le_usr = preprocessing.LabelEncoder()
    le_news = preprocessing.LabelEncoder()
    le_usr.fit(df["usr_id"])
    le_news.fit(df["news_id"])
    df["usr_id"] = le_usr.transform(df["usr_id"])
    df["news_id"] = le_news.transform(df["news_id"])
    for row in df.itertuples(index=True, name='Pandas'):
        matrix[row[1], row[2]] += 1
    np.save("user_news", matrix)
    return matrix

class RecNews:

    def __init__(self, k_total):
        self.k_total = k_total

    def prepare(self):
        # train user-news matrix
        self.df_train = pd.read_csv('%s/data/%s' % (root_dir, 'train_jieba.csv'), usecols=[0, 1])
        self.df_train_user = ((self.df_train.copy())['usr_id'])
        self.df_train_user.drop_duplicates(keep='first', inplace=True)
        self.df_train_user.sort_values(inplace=True)
        self.df_train_user.reset_index(drop=True, inplace=True)
        self.df_train_user.to_csv('users_id.csv', index=False)
        self.df_train_news = (self.df_train.copy())['news_id']
        self.df_train_news.drop_duplicates(keep='first', inplace=True)
        self.df_train_news.sort_values(inplace=True)
        self.df_train_news.reset_index(drop=True, inplace=True)
        self.df_train_news.to_csv('train_news_id.csv', index=False)
        train_matrix = to_matrix(self.df_train)
        self.df_u = pd.read_csv('train_users_id.csv', names=['id'])
        self.df_u.sort_values(inplace=True, by=['id'])

        # calculate news click history
        train_news_click = train_matrix.sum(axis=0)
        self.df_train_news_click = pd.DataFrame(train_news_click, columns=['click'])
        self.df_train_news_click = pd.concat([self.df_train_news, self.df_train_news_click], axis=1)
        self.df_train_news_click.sort_values(by=['click'], inplace=True, ascending=False)
        self.df_train_news_click.to_csv('train_news_hot.csv', index_label='index')

        return train_matrix

    def nmf(self, train_matrix):
        nmf = NMF(n_components=2)
        user_distribution = nmf.fit_transform(train_matrix)
        item_distribution = nmf.components_
        reconstruct_matrix = np.dot(user_distribution, item_distribution)
        filter_matrix = train_matrix < 1e-8
        filter_known_matrix = train_matrix > 1e-8
        nmf_result = reconstruct_matrix * filter_matrix
        nmf_known_result = reconstruct_matrix * filter_known_matrix
        np.save('nmf_result', nmf_result)
        np.save('nmf_known_result', nmf_known_result)
        # print(nmf_known_result.shape)
        return (nmf_result, nmf_known_result)

    def rec_based_on_title(self, df_train_title, df_test_title):

        corpora_documents = df_train_title['word_title'].values.tolist()
        corpora_documents = [[j for j in i.split(' ')] for i in corpora_documents]

        # 生成字典和向量语料
        dictionary = corpora.Dictionary(corpora_documents)
        dictionary.save('dictionary.txt')  # 保存生成的词典
        # dictionary = corpora.Dictionary.load('dictionary.txt')#加载
        # 得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
        corpus = [dictionary.doc2bow(text) for text in corpora_documents]
        # 向量的每一个元素代表了一个word在这篇文档中出现的次数
        corpora.MmCorpus.serialize('corpuse.mm', corpus)  # 保存生成的语料
        corpus=corpora.MmCorpus('corpuse.mm')#加载

        # corpus是一个返回bow向量的迭代器
        tfidf_model = models.TfidfModel(corpus)
        tfidf_model.save('tfidf_model.tfidf')
        # tfidf_model = models.TfidfModel.load("tfidf_model.tfidf")

        # 完成对corpus中出现的每一个特征的Iself.df值的统计工作
        corpus_tfidf = tfidf_model[corpus]
        corpus_tfidf.save("data.tfidf")

        # 计算相似度
        similarity = similarities.MatrixSimilarity(corpus_tfidf)
        similarity.save('similarity.index')
        # similarity = similarities.Similarity.load('similarity.index')

        # rec on test set
        corpora_documents_test = df_test_title['word_title'].values.tolist()
        corpora_documents_test = [[j for j in i.split(' ')] for i in corpora_documents_test]

        corpus_test = [dictionary.doc2bow(text) for text in corpora_documents_test]
        corpus_test_tfidf = tfidf_model[corpus_test]
        corpus_test_tfidf.save("data_test.tfidf")

        similarity.num_best = self.k_total
        test_similarity = [similarity[test] for test in corpus_test_tfidf] # 返回最相似的样本材料,(index_of_document, similarity) tuples

        return test_similarity

    def hit(self, test_matrix, predict_matrix):
        score = np.count_nonzero(test_matrix * predict_matrix)

        return score

    def rec_based_on_hot(self, k_old, k_hot, sim_news_hot):
        rec_list = []
        rec_list_old = self.df_train_news_click.head(k_old)['news_id']  # news_id
        rec_list.extend(rec_list_old)
        # new news
        df_sim_list = self.df_train_news_click.head(k_hot)
        n_new_per_old = int((self.k_total - k_old) / k_hot)
        rec_list_new = []
        for i in range(k_hot-1):
            index = df_sim_list.index[i]    # le 热门新闻
            df_new_on_hot = pd.DataFrame([x for x in sim_news_hot[index][:n_new_per_old]], columns=['index', 'similarity'])
            df_new_on_hot.sort_values(by=['similarity'], inplace=True)
            n_hot = df_new_on_hot.shape[0]
            for l in range(min(n_hot,n_new_per_old)):
                rec_list_new.append(self.df_test_news_id.news_id[df_new_on_hot.index[l]])    # news_id

        i = k_hot-1
        index = df_sim_list.index[i]
        n_new_per_old = self.k_total - k_old - len(rec_list_new)
        df_new_on_hot = pd.DataFrame([x for x in sim_news_hot[index][:n_new_per_old]], columns=['index', 'similarity'])
        df_new_on_hot.sort_values(by=['similarity'], inplace=True)
        for l in range(n_new_per_old):
            rec_list_new.append(self.df_test_news_id.news_id[df_new_on_hot.index[l]])  # news_id
        rec_list.extend(rec_list_new)

        return rec_list

    def recommendation(self, base=1):
        # para:
        #   base: train text data; 1 title, 2 content

        # get train news cut by jieba
        self.df_train_title = pd.read_csv('%s/data/%s' % (root_dir, 'train_jieba.csv'), encoding='utf-8', usecols=[1, base+2])
        self.df_train_title.drop_duplicates(inplace=True, keep='first')
        self.df_train_title.sort_values(by='news_id', inplace=True)
        self.df_train_title.reset_index(inplace=True)
        # get test set
        self.df_test = pd.read_csv('%s/data/%s' % (root_dir, 'test_jieba.csv'), encoding='utf-8')
        self.df_news = pd.read_csv('%s/data/%s' % (root_dir, 'news_jieba.csv'), encoding='utf-8', usecols=[0, base])
        self.df_test_news_id = (self.df_test.copy())['news_id']
        self.df_test_news_id = pd.DataFrame(self.df_test_news_id, columns=['news_id'])
        self.df_test_news_id.drop_duplicates(keep='first', inplace=True)
        self.df_test_news_id.sort_values(by=['news_id'], inplace=True)
        self.df_test_news_id.reset_index(inplace=True, drop=True)
        self.df_test_news_id.to_csv('test_news_id.csv', index=False)
        self.df_test_title = pd.merge(self.df_test_news_id, self.df_news, how='left')

        # rec new news
        # sim_news_rec = self.rec_based_on_title()
        sim_news_hot = self.rec_based_on_title(self.df_test_title, self.df_train_title)

        # predict to all test user
        self.df_test_user = ((self.df_test.copy())['usr_id'])
        self.df_test_user.drop_duplicates(keep='first', inplace=True)
        self.df_test_user.sort_values(inplace=True)
        self.df_test_user.reset_index(drop=True, inplace=True)
        self.df_test_user.to_csv('test_users_id.csv', index=False)
        n_test_user = self.df_test_user.shape[0]
        n_test_news = self.df_test.news_id.unique().shape[0]
        predict_matrix = np.zeros((n_test_user, n_test_news))

        # nmf
        train_matrix = self.prepare()
        (nmf_result, nmf_known_result) = self.nmf(train_matrix)
        # rec based on hot
        rec_list_hot = self.rec_based_on_hot(6, 3, sim_news_hot)

        # df_predict = pd.DataFrame()
        for i in range(n_test_user):
            rec_list = []
            index_user = self.df_train_user[self.df_train_user == self.df_test_user[i]]     # series: the index of fit samples in train
            if not index_user.empty:    # old user
                index_user = index_user.reset_index()['index'][0]   # le in train
                k_old = 6
                k_hot = 3
                rec_list_old = []
                rec_list_old_score = nmf_result[index_user]  # nmf score
                n_cf = len(rec_list_old_score)
                df_rec_list_old = pd.DataFrame(rec_list_old_score, columns=['similarity'])
                df_rec_list_old['index'] = list(range(df_rec_list_old.shape[0]))
                df_rec_list_old.sort_values(by=['similarity'], inplace=True, ascending=False)
                for j in range(min(n_cf, k_old)):  # k_old篇旧文章
                    rec_list_old.append(self.df_train_news[df_rec_list_old.index[j]])  # le to news_id
                rec_list.extend(rec_list_old)
                # new news
                rec_list_new = []
                rec_list_new_score = nmf_known_result[index_user]    # nmf score
                n_clicked = len(rec_list_new_score)
                df_rec_list_new = pd.DataFrame(rec_list_new_score, columns=['similarity'])
                df_rec_list_new['index'] = list(range(df_rec_list_new.shape[0]))        # add le
                # print(df_rec_list_new.shape[0])
                df_rec_list_new.sort_values(by=['similarity'], inplace=True, ascending=False)
                n_new_per_old = int((self.k_total - k_old) / k_hot)     # num of rec new news per old news
                for j in range(min(n_clicked, k_hot)):  # k_hot篇已读旧文章
                    # print('%d:%d'%(min(n_clicked, k_hot), j))
                    # print('%d:%d'%(df_rec_list_new.index[j], len(sim_news_hot)))  # error: 4934:4897
                    new_on_old_score = sim_news_hot[df_rec_list_new.index[j]]    # [(le, score)]
                    n_sim = len(new_on_old_score)
                    df_new_on_old = pd.DataFrame(new_on_old_score, columns=['index', 'similarity'])
                    df_new_on_old.sort_values(by='similarity', ascending=False, inplace=True)
                    for l in range(min(n_sim, n_new_per_old)):  # 每篇旧文章，推荐新文章
                        rec_list_new.append(self.df_train_news[df_new_on_old.index[l]])
                # j = k_hot - 1
                # new_on_old_score = sim_news_hot[df_rec_list_new.index[j]]   # [(le, score)]
                # n_sim = len(new_on_old_score)
                # df_new_on_old = pd.DataFrame(new_on_old_score, columns=['index', 'similarity'])
                # df_new_on_old.sort_values(by='similarity', ascending=False, inplace=True)
                # n_new_per_old = self.k_total - k_old - len(rec_list_new)
                # for l in range(min(n_sim, n_new_per_old)):  # 最后一篇旧文章，推荐新文章
                #     rec_list_new.append(self.df_test_news_id.news_id[df_new_on_old.index[l]])   # news_id in test
                rec_list.extend(rec_list_new)

                # record to matrix
                for id in rec_list:
                    index = self.df_test_news_id[self.df_test_news_id['news_id'] == id]
                    if not index.empty:
                        index = index.reset_index()['index'][0]
                        predict_matrix[i][index] += 1

            else:   # new user
                rec_list = rec_list_hot
                for id in rec_list:
                    index = self.df_test_news_id[self.df_test_news_id['news_id'] == id]
                    if not index.empty:
                        index = index.reset_index()['index'][0]
                        predict_matrix[i][index] += 1


        # test matrix
        test_matrix = to_matrix(self.df_test)

        score = self.hit(test_matrix, predict_matrix)
        return score


if __name__ == "__main__":
    # df_data = pd.read_csv('%s/data/%s' % (root_dir, 'user_click_data.txt'), encoding='utf-8', delimiter="\t",
    #                            names=["user_id", "news_id", "scan_time", "news_title", "content", "publish_time"],
    #                            index_col=False)
    # prepare user-news matrix
    # prepare()
    # train_matrix = np.load('user_news.npy')
    # print (train_matrix)

    # NMF
    # nmf_rec = nmf(train_matrix)

    # news similarity based on title
    # rec_based_on_title()

    # new user based on news similar to hot news
    # recommendation()
    rec = RecNews(20)
    print(rec.recommendation())

