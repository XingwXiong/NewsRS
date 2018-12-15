from sklearn.decomposition import NMF
import pandas as pd
import numpy as np


root_dir = '../../'

# gen train user-news matrix
df_train = pd.read_csv('%s/data/%s' % (root_dir, 'train_click.csv'), usecols=[0, 1, 2])
train_usrid = df_train['usr_id'].drop_duplicates()
train_newsid = df_train['news_id'].drop_duplicates()

data = pd.pivot_table(df_train, index=['usr_id'], values=['click'], columns=['news_id'], aggfunc=[np.sum], fill_value=0)
train_matrix = data.values


nmf = NMF(n_components=2)
user_distribution = nmf.fit_transform(train_matrix)
item_distribution = nmf.components_
reconstruct_matrix = np.dot(user_distribution, item_distribution)
filter_matrix = train_matrix < 1e-8
nmf_result = reconstruct_matrix * filter_matrix


def topkUsers(news_id, k):
    # find corres news index
    news_id_index = pd.Index(train_newsid)
    idx = news_id_index.get_loc(news_id)
    print('news_id index:{}'.format(idx))

    ratings = list(nmf_result[:, idx])
    indexs = [i for i in range(ratings.__len__())]
    index_ratings = list(zip(indexs, ratings))
    index_ratings.sort(key=lambda x: x[1], reverse=True)
    usr_idxs = [tup[0] for tup in index_ratings[:k]]
    # find corres users
    users = list(train_usrid.iloc[usr_idxs])
    return users


def topkNews(usr_id, k):
    # find corres news index
    usr_id_index = pd.Index(train_usrid)
    idx = usr_id_index.get_loc(usr_id)

    ratings = list(nmf_result[idx, :])
    indexs = [i for i in range(ratings.__len__())]
    index_ratings = list(zip(indexs, ratings))
    index_ratings.sort(key=lambda x: x[1], reverse=True)
    news_idxs = [tup[0] for tup in index_ratings[:k]]
    # find corres users
    news_ids = list(train_newsid.iloc[news_idxs])
    return news_ids











