import os
import math
import numpy as np
import pickle
import random
import pandas as pd
import pickle
import jieba
jieba.set_dictionary('dict.txt.big')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('podcast_episode_data.pickle', 'rb+') as file:
    dataset = pickle.load(file)

def get_term(text):
    stopWords=[]
    segments=[]

    with open('stopWords.txt', 'r', encoding='UTF-8') as file:
        for data in file.readlines():
            data = data.strip()
            stopWords.append(data)

    segments = jieba.cut(text, cut_all=False, HMM=True)
    remainderWords = list(filter(lambda a: a not in stopWords and a != '\n' and a != ' ', segments))
    return ' '.join(remainderWords)

def get_tfidf(d_list):
    for key in ['show_name', 'episode_name', 'show_description', 'episode_description']:
        corpus = []
        for d in d_list:
            corpus.append(d[key])
        print(f"NOW is {key}.")
        vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
        vectoerizer.fit(corpus)
        X = vectoerizer.transform(corpus)
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(X.toarray())
        tfidf = tfidf_transformer.transform(X)
        
        for res, d in zip(tfidf, d_list):
            d[key] = res.toarray()
    
    print("return")
    return d_list

for i in range(len(dataset)):
    dataset[i]['show_name'] = get_term(dataset[i]['show_name'])
    dataset[i]['episode_name'] = get_term(dataset[i]['episode_name'])
    dataset[i]['show_description'] = get_term(dataset[i]['show_description'])
    dataset[i]['episode_description'] = get_term(dataset[i]['episode_description'])

tf_idf_dataset = get_tfidf(dataset)
print('TF iDF Done !')

table_dict = {}
for key in tf_idf_dataset[0].keys():
    table_dict[key] = pd.DataFrame()
    print(key)
    if key in ['show_name', 'episode_name', 'show_description', 'episode_description']:
        temp = []
        for d in tf_idf_dataset:
            temp.append(d[key][0].tolist())
        
        table_dict[key] = pd.DataFrame(cosine_similarity(np.array(temp, dtype=float)))  
        
    elif key not in ['img']:
        total = []
        for i in tf_idf_dataset:
            single = []
            for j in tf_idf_dataset:
                if i[key] == j[key]:
                    single.append(1)
                else:
                    single.append(0)
            total.append(single)
            
        table_dict[key] = pd.DataFrame(np.array(total).T)

with open('tf_idf_similarity.pickle', 'wb') as handle:
    pickle.dump(table_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)