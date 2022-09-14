# Podcast-Recommender
Recommend similar podcast shows to the current one.

## Usage

### Preprocessing
**Run the preprocessing.py**
```
python3 preprocessing.py
```
**or**
**Download the pickle file on the GD for getting the similarity table**
```
bash Download.sh
```
** Each similarity pickle file gets 1.4GB **
---

### Running
**Run the command**
```
python3 main.py
```
**Simply check it on the browser, it's on 127.0.0.1:5000**

## Demo
### You can choose two ways of embedding, tf-idf or Bert embedding.
![img1](https://github.com/qwnim2/Podcast-Recommender/blob/master/demo%20img/%E6%88%AA%E5%9C%96%202021-01-20%20%E4%B8%8A%E5%8D%884.33.36.png)
---
### The one you are now listening to...
![img2](https://github.com/qwnim2/Podcast-Recommender/blob/master/demo%20img/%E6%88%AA%E5%9C%96%202021-01-20%20%E4%B8%8A%E5%8D%884.35.07.png)
---
## The recommend podcast show corresponding to the current one.
![img3](https://github.com/qwnim2/Podcast-Recommender/blob/master/demo%20img/%E6%88%AA%E5%9C%96%202021-01-20%20%E4%B8%8A%E5%8D%884.35.41.png)
---
## You can change to Bert embedding and find the difference
![img4](https://github.com/qwnim2/Podcast-Recommender/blob/master/demo%20img/%E6%88%AA%E5%9C%96%202021-01-20%20%E4%B8%8A%E5%8D%884.36.22.png)
