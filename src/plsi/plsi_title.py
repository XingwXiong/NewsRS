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
    return matrix

class RecNews:

    def __init__(self, k_total, k_old_new, k_hot_new, k_old, k_hot, base):
        self.k_total = k_total
        self.k_old_new = k_old_new
        self.k_hot_new = min(k_hot_new, self.k_total-k_old_new)
        self.k_old = k_old
        self.k_hot = min(k_old, self.k_total - k_old)
        self.base = base    # 1-title 2-content
        if base == 1:
            self.name = 'word_title'
        elif base == 2:
            self.name = 'word_content'

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
        np.save("train_user_news", train_matrix)

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
        return (nmf_result, nmf_known_result)

    def rec_based_on_title(self, df_train_title, df_test_title):

        corpora_documents = df_train_title[self.name].values.tolist()
        corpora_documents = [[j for j in i.split(' ')] for i in corpora_documents]

        dictionary = corpora.Dictionary(corpora_documents)
        dictionary.save('dictionary.txt')  # 保存生成的词典
        # dictionary = corpora.Dictionary.load('dictionary.txt')#加载
        corpus = [dictionary.doc2bow(text) for text in corpora_documents]
        corpora.MmCorpus.serialize('corpuse.mm', corpus)  # 保存生成的语料
        corpus=corpora.MmCorpus('corpuse.mm')#加载

        tfidf_model = models.TfidfModel(corpus)
        tfidf_model.save('tfidf_model.tfidf')
        # tfidf_model = models.TfidfModel.load("tfidf_model.tfidf")

        corpus_tfidf = tfidf_model[corpus]
        corpus_tfidf.save("data.tfidf")

        similarity = similarities.MatrixSimilarity(corpus_tfidf)
        similarity.save('similarity.index')
        # similarity = similarities.Similarity.load('similarity.index')

        # rec on test set
        corpora_documents_test = df_test_title[self.name].values.tolist()
        corpora_documents_test = [[j for j in i.split(' ')] for i in corpora_documents_test]

        corpus_test = [dictionary.doc2bow(text) for text in corpora_documents_test]
        corpus_test_tfidf = tfidf_model[corpus_test]
        corpus_test_tfidf.save("data_test.tfidf")

        similarity.num_best = self.k_total
        test_similarity = [similarity[test] for test in corpus_test_tfidf] # 返回最相似的样本材料,(index_of_document, similarity) tuples

        return test_similarity

    def hit(self, test_matrix, predict_matrix):
        print('max:{},mean:{}'.format(predict_matrix.max(),predict_matrix.mean()))
        score = np.count_nonzero(test_matrix * predict_matrix)
        return score

    def rec_based_on_hot(self, sim_news_hot):
        rec_list = []
        rec_list_old = self.df_train_news_click.head(self.k_old_new)['news_id']  # news_id
        rec_list.extend(rec_list_old)
        # new news
        df_sim_list = self.df_train_news_click.head(self.k_hot_new)
        n_new_per_old = int((self.k_total - self.k_old_new) / self.k_hot_new)
        rec_list_new = []
        for i in range(self.k_hot_new-1):
            index = df_sim_list.index[i]    # le 热门新闻
            df_new_on_hot = pd.DataFrame([x for x in sim_news_hot[index][:n_new_per_old]], columns=['index', 'similarity'])
            df_new_on_hot.sort_values(by=['similarity'], inplace=True)
            n_hot = df_new_on_hot.shape[0]
            for l in range(min(n_hot,n_new_per_old)):
                rec_list_new.append(self.df_test_news_id.news_id[df_new_on_hot.index[l]])    # news_id

        i = self.k_hot_new-1
        index = df_sim_list.index[i]
        n_new_per_old = self.k_total - self.k_old_new - len(rec_list_new)
        df_new_on_hot = pd.DataFrame([x for x in sim_news_hot[index][:n_new_per_old]], columns=['index', 'similarity'])
        df_new_on_hot.sort_values(by=['similarity'], inplace=True)
        for l in range(n_new_per_old):
            rec_list_new.append(self.df_test_news_id.news_id[df_new_on_hot.index[l]])  # news_id
        rec_list.extend(rec_list_new)

        return rec_list

    def train(self):

        # get train news cut by jieba
        self.df_train_title = pd.read_csv('%s/data/%s' % (root_dir, 'train_jieba.csv'), encoding='utf-8',
                                          usecols=[1, self.base + 2])
        self.df_train_title.drop_duplicates(inplace=True, keep='first')
        self.df_train_title.sort_values(by='news_id', inplace=True)
        self.df_train_title.reset_index(inplace=True,drop=True)
        self.df_train_title.to_csv('train_title.csv', index=False)

        # nmf data preparation
        self.train_matrix = self.prepare()

    def recommendation(self):

        self.train()

        # get test set
        self.df_test = pd.read_csv('%s/data/%s' % (root_dir, 'test_jieba.csv'), encoding='utf-8')
        self.df_news = pd.read_csv('%s/data/%s' % (root_dir, 'news_jieba.csv'), encoding='utf-8', usecols=[0, self.base])
        self.df_test_news_id = (self.df_test.copy())['news_id']
        self.df_test_news_id = pd.DataFrame(self.df_test_news_id, columns=['news_id'])
        self.df_test_news_id.drop_duplicates(keep='first', inplace=True)
        self.df_test_news_id.sort_values(by=['news_id'], inplace=True)
        self.df_test_news_id.reset_index(inplace=True, drop=True)
        self.df_test_news_id.to_csv('test_news_id.csv', index=False)
        self.df_test_title = pd.merge(self.df_test_news_id, self.df_news, how='left')

        # matrix: all test user - all test news
        self.df_test_user = ((self.df_test.copy())['usr_id'])
        self.df_test_user.drop_duplicates(keep='first', inplace=True)
        self.df_test_user.sort_values(inplace=True)
        self.df_test_user.reset_index(drop=True, inplace=True)
        self.df_test_user.to_csv('test_users_id.csv', index=False)
        n_test_user = self.df_test_user.shape[0]
        n_test_news = self.df_test.news_id.unique().shape[0]
        predict_matrix = np.zeros((n_test_user, n_test_news))

        # nmf
        self.train_matrix = np.load('train_user_news.npy')
        (nmf_result, nmf_known_result) = self.nmf(self.train_matrix)

        # similarity based on title
        self.df_train_title = pd.read_csv('train_title.csv', encoding='utf-8')
        # sim_news_rec = self.rec_based_on_title()
        sim_news_hot = self.rec_based_on_title(self.df_test_title, self.df_train_title)

        # rec based on hot
        self.df_train_news_click = pd.read_csv('train_news_hot.csv')
        rec_list_hot = self.rec_based_on_hot(sim_news_hot)

        # begin to recommend to all test users
        for i in range(n_test_user):
            rec_list = []
            index_user = self.df_train_user[self.df_train_user == self.df_test_user[i]]     # series: the index of fit samples in train

            if not index_user.empty:    # old user
                index_user = index_user.reset_index()['index'][0]   # le in train
                # old news
                rec_list_old = []
                rec_list_old_score = nmf_result[index_user]  # nmf score
                n_cf = len(rec_list_old_score)
                df_rec_list_old = pd.DataFrame(rec_list_old_score, columns=['similarity'])
                df_rec_list_old['index'] = list(range(df_rec_list_old.shape[0]))
                df_rec_list_old.sort_values(by=['similarity'], inplace=True, ascending=False)
                for j in range(min(n_cf, self.k_old)):  # k_old篇旧文章
                    rec_list_old.append(self.df_train_news[df_rec_list_old.index[j]])  # le to news_id
                rec_list.extend(rec_list_old)

                # new news
                rec_list_new = []
                rec_list_new_score = nmf_known_result[index_user]    # nmf score
                n_clicked = len(rec_list_new_score)
                df_rec_list_new = pd.DataFrame(rec_list_new_score, columns=['similarity'])
                df_rec_list_new['index'] = list(range(df_rec_list_new.shape[0]))        # add le
                df_rec_list_new.sort_values(by=['similarity'], inplace=True, ascending=False)
                n_new_per_old = int((self.k_total - self.k_old) / self.k_hot)     # num of rec new news per old news
                for j in range(min(n_clicked, self.k_hot)):  # k_hot篇已读旧文章
                    new_on_old_score = sim_news_hot[df_rec_list_new.index[j]]    # [(le, score)]
                    n_sim = len(new_on_old_score)
                    df_new_on_old = pd.DataFrame(new_on_old_score, columns=['index', 'similarity'])
                    df_new_on_old.sort_values(by='similarity', ascending=False, inplace=True)
                    for l in range(min(n_sim, n_new_per_old)):  # 每篇旧文章，推荐新文章
                        rec_list_new.append(self.df_train_news[df_new_on_old.index[l]])
                j = self.k_hot - 1
                new_on_old_score = sim_news_hot[df_rec_list_new.index[j]]   # [(le, score)]
                n_sim = len(new_on_old_score)
                df_new_on_old = pd.DataFrame(new_on_old_score, columns=['index', 'similarity'])
                df_new_on_old.sort_values(by='similarity', ascending=False, inplace=True)
                n_new_per_old = self.k_total - self.k_old - len(rec_list_new)
                for l in range(min(n_sim, n_new_per_old)):  # 最后一篇旧文章，推荐新文章
                    rec_list_new.append(self.df_test_news_id.news_id[df_new_on_old.index[l]])   # news_id in test
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

    rec = RecNews(10, 5,2,6,1,1)
    print(rec.recommendation())

