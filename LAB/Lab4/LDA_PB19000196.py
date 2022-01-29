import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


class LDA(object):
    def __init__(self,
                 alpha=1 / 8,
                 eta=1 / 8,
                 k=8,  # number of topics
                 itr=50,
                 seed=None
                 ):
        self.alpha = alpha
        self.eta = eta
        self.K = k
        self.itr = itr
        self.seed = seed
        self.theta = None  # doc_topic_distr
        self.beta = None  # topic_word_distr
        self.ndz = None
        self.nzw = None
        self.nz = None
        self.Z = None  # (d,w),文章d的单词w对应话题

    def init(self, X):  # random init matrix
        self.Z = []
        N, M = X.shape
        self.ndz = np.zeros([N, self.K]) + self.alpha
        self.nzw = np.zeros([self.K, M]) + self.eta
        self.nz = np.zeros([self.K]) + M * self.eta
        for d, fword in enumerate(X):  # fword:每篇文章的词频
            doc_zlist = []
            for w, f in enumerate(fword):
                if f != 0:
                    for _ in range(f):
                        pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                        z = np.random.multinomial(1, pz / pz.sum()).argmax()  # 随机选择
                        doc_zlist.append(z)
                        self.ndz[d, z] += 1
                        self.nzw[z, w] += 1
                        self.nz[z] += 1
            self.Z.append(doc_zlist)

    def gibbsSampling(self, X):
        if self.Z is None:
            print("you need to init first!")
        # 为每个文档中的每个单词重新采样topic
        for d, fword in enumerate(X):
            index = 0
            for w, f in enumerate(fword):
                if f != 0:
                    for _ in range(f):
                        z = self.Z[d][index]
                        # 除去自身
                        self.ndz[d, z] -= 1
                        self.nzw[z, w] -= 1
                        self.nz[z] -= 1
                        # 重新计算当前文档当前单词属于每个topic的概率
                        pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                        # 按照计算出的分布进行采样
                        z = np.random.multinomial(1, pz / pz.sum()).argmax()
                        self.Z[d][index] = z
                        # 还原
                        self.ndz[d, z] += 1
                        self.nzw[z, w] += 1
                        self.nz[z] += 1
                        index += 1

    def fit(self, X):
        if self.seed is not None:
            np.random.seed(self.seed)
        print("initializing...")
        self.init(X)
        print("init finished...")
        print("training begin:")
        for i in range(self.itr):
            self.gibbsSampling(X)
            print("Iteration: ", i + 1, " Completed", )


def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    # jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        # word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)
        word = seg_word.word
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # this word is stopword
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    return (" ").join(word_list)


# def preprocess(data_path, stop_dic_path, dic_path):
#     data = pd.read_excel(data_path)  # content type
#     file = codecs.open(stop_dic_path, 'r', 'utf-8')
#     stopwords = [line.strip() for line in file]
#     file.close()
#     # cut docs
#     content = data["content"]
#     documents = []
#     for i in range(data.shape[0]):
#         documents.append(chinese_word_cut(content[i], dic_path))
#     # word <->index
#     word2id = {}
#     id2word = {}
#     docs = []
#     currentDocument = []
#     currentWordId = 0
#     for document in documents:
#         for word in document:
#             word = word.lower().strip()
#             # 单词长度大于1并且不包含数字并且不是停止词
#             if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
#                 if word in word2id:
#                     currentDocument.append(word2id[word])
#                 else:
#                     currentDocument.append(currentWordId)
#                     word2id[word] = currentWordId
#                     id2word[currentWordId] = word
#                     currentWordId += 1
#         docs.append(currentDocument)
#         currentDocument = []
#     return docs, word2id, id2word


if __name__ == '__main__':
    data_path = "./data/data.xlsx"
    dic_file = "./data/dict.txt"
    stop_file = "./data/stopwords.txt"
    data = pd.read_excel("./data/data.xlsx")  # content type
    # docs, word2id, id2word = preprocess(data_path, stop_file, dic_file)
    # print(docs[0])
    # print(word2id)
    # print(id2word)
    data["content_cutted"] = data.content.apply(chinese_word_cut)
    n_features = 1000  # 提取1000个特征词语
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10)
    tf = tf_vectorizer.fit_transform(data.content_cutted)
    doc_word_mat = tf.toarray()
    mylda = LDA(k=8, seed=0, itr=50)
    mylda.fit(doc_word_mat)
    n_top_words = 15
    tf_feature_names = tf_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(mylda.nzw):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(topic_w)
