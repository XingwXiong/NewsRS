import pandas as pd
from gensim import corpora, models, similarities
import nmf


root_dir = '../../'
# concat jieba to gen news
train_data = pd.read_csv('%s/data/%s' % (root_dir, 'train_jieba.csv'), encoding='utf-8')
# gen corpus
train_news = train_data[['news_id', 'word_title']]
train_titles = train_news['word_title'].tolist()
train_users = train_data['usr_id'].tolist()
# gen tf-idf
stoplist = []
texts = [[word for word in title.split() if word not in stoplist] for title in train_titles]
dict = corpora.Dictionary(texts)
dict.save('dictionary.txt')
corpus = [dict.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
tfidf.save('%s/model/%s' % (root_dir, 'tdidf'))
corpus_tfidf = tfidf[corpus]
index = similarities.MatrixSimilarity(corpus_tfidf)
# gen cosine similarity matrix
test_data = pd.read_csv('%s/data/%s' % (root_dir, 'test_jieba.csv'), encoding='utf-8')
test_news = test_data['word_title']
test_titles = test_news.tolist()
test_users = test_data['usr_id'].tolist()

# gen usr-item rate matrix
train_click = pd.read_csv('%s/data/%s' % (root_dir, 'train_click.csv'))
sorted_train_click = train_click.sort_index(by='click', ascending=False)


def findMaxSimilarNews(title):
    vec = dict.doc2bow(title.split())
    vec_tdidf = tfidf[vec]
    # compare vec with corpus
    sims = list(index[vec_tdidf])
    maxidx = sims.index(max(sims))
    # print('max similar title index:{}'.format(maxidx))
    return maxidx


def topkUsers(news_id, k):
    temp = sorted_train_click[sorted_train_click['news_id'] == news_id]
    users = temp['usr_id']
    return users.tolist()[:k]


# count hit
def calculateHit(train_click_dict, testcsv):
    hit = 0
    test_data = pd.read_csv(testcsv)
    for key in train_click_dict:
        usr_id = int(key[0])
        news_id = int(key[1])
        click1 = int(train_click_dict[key])
        df = test_data[(test_data['usr_id'] == usr_id) & (test_data['news_id'] == news_id)]
        if not df.empty:
            click2 = df.reset_index()['click'][0]
            hit = hit + min(click1, click2)
    print('hit:{}'.format(hit))
    return hit


if __name__ == '__main__':
    train_click_dict = {}
    test_titles = list(set(test_titles))
    new_titles = set(test_titles).difference(set(train_titles))
    print(new_titles.__len__())
    for idx, title in enumerate(new_titles):
        maxidx = findMaxSimilarNews(title)
        #find corres newsid
        newsid = train_data.loc[maxidx, 'news_id']
        print('idx:{},news_id:{}'.format(idx, newsid))

        users = topkUsers(newsid, k=10)

        for usr in users:
            key = (usr, newsid)
            if key not in train_click_dict:
                train_click_dict[key] = 1
            else:
                train_click_dict[key] = train_click_dict[key] + 1
    # calculate hit
    testcsv = '%s/data/%s' % (root_dir, 'test_click.csv')
    hit1 = calculateHit(train_click_dict, testcsv)

    train_click_dict = {}
    old_users = list(set(test_users).intersection(set(train_users)))

    for idx, usr in enumerate(old_users):
        print('idx: {}, usr_id: {}'.format(idx, usr))
        news_ids = nmf.topkNews(usr, k=10)
        for news_id in news_ids:
            key = (usr, news_id)
            if key not in train_click_dict:
                train_click_dict[key] = 1
            else:
                train_click_dict[key] = train_click_dict[key] + 1
    # calculate hit
    testcsv = '%s/data/%s' % (root_dir, 'test_click.csv')
    hit2 = calculateHit(train_click_dict, testcsv)
    print(hit1+hit2)






