import numpy as np
import pickle
import random
import pandas as pd
import pickle
import random

def arrWeight(weight, table_dict, nextId, lastId):
    weightGrad = []
    for key in table_dict.keys():
        if key != 'img':
#             print(key)
#             print(table_dict[key].values[init_ind].shape)
            weightGrad.append(table_dict[key].values[lastId][nextId])
    
    weightGrad = np.array(weightGrad)
#     print(weightGrad, weight[0])
    for i in range(weightGrad.shape[0]):
        weightGrad[i] = weightGrad[i]*weight[0][i]
    
    weightGrad = weightGrad/np.sum(weightGrad)
#     print(weightGrad)
    weight = (weight + weightGrad)
    weight = weight*100/np.sum((weight))
#     print(weight)
    return(weight)

def makeRecommend(lastId, weight, dataset, n_recommand, max_sameShow):
    X = []
#     print(table_dict['genre'].shape)
    for key in table_dict.keys():
        if key != 'img':
            X.append(table_dict[key].values[lastId].tolist())
    X = np.array(X)
    shit = weight.dot(X)
    candidates = shit[0].argsort()[-100:][::-1]
#     print(candidates)
    
    recommend_list = []
    picked_episode_name = []
    recommend_cnt = 0
    sameShow_cnt = 0
    for candi in candidates:
        if dataset[lastId]['episode_name'] == dataset[candi]['episode_name'] or dataset[candi]['episode_name'] in picked_episode_name:
            foo = 'bar'
#             print('重複了！！')
        else:
            if sameShow_cnt<max_sameShow and dataset[lastId]['show_name']==dataset[candi]['show_name']:
                recommend_list.append(candi)
                recommend_cnt += 1
                sameShow_cnt += 1
                picked_episode_name.append(dataset[candi]['episode_name'])
            elif sameShow_cnt<max_sameShow and dataset[lastId]['show_name']!=dataset[candi]['show_name']:
                recommend_list.append(candi)
                recommend_cnt += 1
                picked_episode_name.append(dataset[candi]['episode_name'])
            elif sameShow_cnt>=max_sameShow and dataset[lastId]['show_name']==dataset[candi]['show_name']:
                foo = 'bar'
#                 print('太多了！！')
            elif sameShow_cnt>=max_sameShow and dataset[lastId]['show_name']!=dataset[candi]['show_name']:
                recommend_list.append(candi)
                recommend_cnt += 1
                picked_episode_name.append(dataset[candi]['episode_name'])
                
            if recommend_cnt == n_recommand:
                break
    
#     print(recommend_list)
    return(recommend_list)

def make_recommend(ind):
    X = []
    for key in table_dict.keys():
        if key != 'img':
            X.append(table_dict[key].values[ind].tolist())
    
        #table_dict[key] = table_dict[key].drop(table_dict[key].index[[ind]])

    X = np.array(X)

    scores_arr = weight.dot(X)
    top_six = scores_arr[0].argsort()[-20:-14][::-1]
    
    episode = []
    publisher = []
    img = []
    candidate = []
    for i in top_six:
        print(i)
        episode.append(dataset[i]['episode_name'])
        publisher.append(dataset[i]['publisher'])
        img.append(dataset[i]['img'])
        candidate.append(i)

    return episode, publisher, img, candidate

def bert_make_recommend(ind):
    X = []
    for key in bert_table_dict.keys():
        if key != 'img':
            X.append(bert_table_dict[key].values[ind].tolist())
    
        #table_dict[key] = table_dict[key].drop(table_dict[key].index[[ind]])

    X = np.array(X)

    scores_arr = weight.dot(X)
    top_six = scores_arr[0].argsort()[-20:-14][::-1]
    
    episode = []
    publisher = []
    img = []
    candidate = []
    for i in top_six:
        print(i)
        episode.append(dataset[i]['episode_name'])
        publisher.append(dataset[i]['publisher'])
        img.append(dataset[i]['img'])
        candidate.append(i)

    return episode, publisher, img, candidate