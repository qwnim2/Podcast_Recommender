from flask import Flask, make_response, request
from flask import redirect, render_template, url_for, send_file
import os
import numpy as np
import pickle
import random
import pandas as pd
import pickle
import random     
from utils import *

with open('podcast_episode_data.pickle', 'rb+') as file:
    dataset = pickle.load(file)

with open('tf_idf_similarity.pickle', 'rb') as handle:
    table_dict = pickle.load(handle)

with open('bert_similarity_table.pickle', 'rb') as handle:
    bert_table_dict = pickle.load(handle)

init_ind = random.randint(0,5412)

weight = np.array([[3,5,10,2,10,70]])
# show_name episode_name genre publisher show_description episode_description img

template_dir = os.path.abspath('./templates')
app = Flask(__name__, template_folder=template_dir)

@app.route("/") 
def index():
    return render_template('index.html')

@app.route("/test", methods=["GET", "POST"])
def listening():
    if  not request.args.get('cadidate') is None:
        prev_ind, next_ind = int(request.args.get('prev')), int(request.args.get('cadidate'))
        print(next_ind)
        print("===============")
        global weight
        weight = arrWeight(weight, table_dict, next_ind, prev_ind)
        recommend_list = makeRecommend(next_ind, weight, dataset, n_recommand=6, max_sameShow=3)
        print(recommend_list)
        episode, publisher, img, candidate = [], [], [], []
        for i in recommend_list:
            print(i)
            episode.append(dataset[i]['episode_name'])
            publisher.append(dataset[i]['publisher'])
            img.append(dataset[i]['img'])
            candidate.append(i)

        prev_ind = next_ind
        # top_six_episode,  top_six_publisher, img, candidate = make_recommend(next_ind)
        print(dataset[next_ind]['episode_name'])
        return render_template('listening.html',\
                                podcast_name = dataset[next_ind]['episode_name'],\
                                podcaster = '主持人：' + dataset[next_ind]['publisher'],\
                                top_six = episode,\
                                top_six_publisher = publisher,\
                                six_candidate = candidate,\
                                prev = prev_ind,\
                                discription = dataset[next_ind]['episode_description'],\
                                img = img,\
                                png = dataset[next_ind]['img'])

    top_six_episode,  top_six_publisher, img, candidate = make_recommend(init_ind)    
    return render_template('listening.html',\
                            podcast_name = dataset[init_ind]['episode_name'],\
                            podcaster = '主持人：' + dataset[init_ind]['publisher'],\
                            top_six = top_six_episode,\
                            top_six_publisher = top_six_publisher,\
                            six_candidate = candidate,\
                            prev = init_ind,\
                            discription = dataset[init_ind]['episode_description'],\
                            img = img,\
                            png = dataset[init_ind]['img'])

@app.route("/bert", methods=["GET", "POST"])
def bert():
    if  not request.args.get('cadidate') is None:
        prev_ind, next_ind = int(request.args.get('prev')), int(request.args.get('cadidate'))
        print(next_ind)
        print("===============")
        global weight
        weight = arrWeight(weight, bert_table_dict, next_ind, prev_ind)
        recommend_list = makeRecommend(next_ind, weight, dataset, n_recommand=6, max_sameShow=3)
        print(recommend_list)
        episode, publisher, img, candidate = [], [], [], []
        for i in recommend_list:
            print(i)
            episode.append(dataset[i]['episode_name'])
            publisher.append(dataset[i]['publisher'])
            img.append(dataset[i]['img'])
            candidate.append(i)

        prev_ind = next_ind
        # top_six_episode,  top_six_publisher, img, candidate = make_recommend(next_ind)
        print(dataset[next_ind]['episode_name'])
        return render_template('bert.html',\
                                podcast_name = dataset[next_ind]['episode_name'],\
                                podcaster = '主持人：' + dataset[next_ind]['publisher'],\
                                top_six = episode,\
                                top_six_publisher = publisher,\
                                six_candidate = candidate,\
                                prev = prev_ind,\
                                discription = dataset[next_ind]['episode_description'],\
                                img = img,\
                                png = dataset[next_ind]['img'])

    top_six_episode,  top_six_publisher, img, candidate = bert_make_recommend(init_ind)    
    return render_template('bert.html',\
                            podcast_name = dataset[init_ind]['episode_name'],\
                            podcaster = '主持人：' + dataset[init_ind]['publisher'],\
                            top_six = top_six_episode,\
                            top_six_publisher = top_six_publisher,\
                            six_candidate = candidate,\
                            prev = init_ind,\
                            discription = dataset[init_ind]['episode_description'],\
                            img = img,\
                            png = dataset[init_ind]['img'])

if __name__ == "__main__":
    app.debug = True
    app.run()