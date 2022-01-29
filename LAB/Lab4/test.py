import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


# output_path = 'D:/python/lda/result'
#file_path = 'D:/python/lda/data'
#os.chdir(file_path)
data = pd.read_excel("./data/data.xlsx")  # content type
# os.chdir(output_path)
dic_file = "./data/dict.txt"
stop_file = "./data/stopwords.txt"

# def chinese_word_cut(mytext):
#     jieba.load_userdict(dic_file)
#     jieba.initialize()
#     try:
#         stopword_list = open(stop_file, encoding='utf-8')
#     except:
#         stopword_list = []
#         print("error in stop_file")
#     stop_list = []
#     flag_list = ['n', 'nz', 'vn']
#     for line in stopword_list:
#         line = re.sub(u'\n|\\r', '', line)
#         stop_list.append(line)
#
#     word_list = []
#     # jieba分词
#     seg_list = psg.cut(mytext)
#     for seg_word in seg_list:
#         # word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)
#         word = seg_word.word
#         find = 0
#         for stop_word in stop_list:
#             if stop_word == word or len(word) < 2:  # this word is stopword
#                 find = 1
#                 break
#         if find == 0 and seg_word.flag in flag_list:
#             word_list.append(word)
#     return (" ").join(word_list)
#
#
# data["content_cutted"] = data.content.apply(chinese_word_cut)
# n_features = 1000  # 提取1000个特征词语
# tf_vectorizer = CountVectorizer(strip_accents='unicode',
#                                 max_features=n_features,
#                                 stop_words='english',
#                                 max_df=0.5,
#                                 min_df=10)
# tf = tf_vectorizer.fit_transform(data.content_cutted) #(800,1000)
# print('hello')
a=[[1,2,3,4,5],[2,3,4,5,6]]
for i,x in enumerate(np.array(a)):
    print(x)