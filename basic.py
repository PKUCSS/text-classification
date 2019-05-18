# 使用TF-IDF特征对Baseline进型改进 ; 训练词向量，并用词向量取平均的方法计算文本相似度

import pickle
import jieba
import numpy as np
from nltk.corpus import stopwords 
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import jieba
import gensim
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import * 
from sklearn.metrics import confusion_matrix
import pandas as pd    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier   



# 读入数据 

label2id = {}
def split_words(dataset):
    raw_docs = []
    docs = []
    labels = []
    for topic, datas in dataset.items():
        if not topic in label2id.keys():
            label2id[topic] = len(list(label2id))
        for data in datas:
            seg = jieba.cut(data["title"])
            raw_docs.append([topic, data["title"]])
            docs.append(" ".join(seg))
            labels.append(label2id[topic])
    return raw_docs, docs, labels

with open("dataset/train.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("dataset/valid.pkl", "rb") as f:
    valid_data = pickle.load(f)
    

train_raw, train_docs, train_labels = split_words(train_data)
valid_raw, valid_docs, valid_labels = split_words(valid_data)  

# 读入停用词表 

stopwords = open("stopwords.txt", "r",encoding="utf-8").readlines()
stopwords = [i.strip('\n') for i in stopwords]
print(stopwords[:100]) 

# 使用TfidfVectorizer改进词袋子，准确率提升至82.9% 

bowModel = TfidfVectorizer(stop_words=stopwords).fit(train_docs)
train_x = bowModel.transform(train_docs)
valid_x = bowModel.transform(valid_docs)
model = MultinomialNB() 
model.fit(train_x, train_labels)

prediction = model.predict(valid_x)
print('acc = %.4f' % (sum(prediction == valid_labels) / len(valid_labels)))

print(pd.DataFrame(confusion_matrix(valid_labels, prediction)))

# 寻找最相似新闻的代码

def Find(query_str):
    seg = jieba.cut(query_str)
    vec = bowModel.transform([" ".join(seg)])
    score = np.zeros(train_x.shape[0])
    for i in range(train_x.shape[0]):
        diff = np.array(vec) - np.array(train_x[i])
        score[i] = abs(diff).sum()
    ids = list(range(train_x.shape[0]))
    ids.sort(key=lambda x:score[x])
    for i in ids[:20]:
        print(train_raw[i][0], train_raw[i][1])
    print("\n")

Find("王者荣耀国际版入选东南亚运动会电竞项目")
Find("首只独角兽周二上市！8个涨停赚4万")       
Find("名爵EZS上市最低仅需9.99万元 高颜值更靠谱的纯电动SUV") 

# 训练词向量模型并保存 
import gensim 
self_train_model = gensim.models.Word2Vec([doc.split() for doc in train_docs]+[doc.split() for doc in valid_docs] ,min_count=1,sg=1,size=300,iter=50)    
self_train_model.wv.save_word2vec_format('train_result.txt')
self_train_model.most_similar("北大")  


# 用词向量取平均，将句子向量化
import pandas as pd 
new_train_cv,new_valid_cv = [],[]
for sentence in train_docs:
    sentence = sentence.split()
    vec = np.zeros(300)
    for word in sentence:  
        if word in self_train_model.wv.vocab :    
            vec += self_train_model[word]*1/len(sentence)
        else:
            vec += np.zeros(300)
    #print(vec)
    new_train_cv.append(vec)
new_valid_cv = []
for sentence in valid_docs:
    sentence = sentence.split()
    vec = np.zeros(300)
    for word in sentence:
        if word in self_train_model.wv.vocab:     
            vec += self_train_model[word]*1/len(sentence)
        else:
            vec += np.zeros(300)
    #print(vec)
    new_valid_cv.append(vec)
    
# 测试分类准确度，达68%左右     
newmodel = GaussianNB()
newmodel.fit(new_train_cv, train_labels) 

prediction = newmodel.predict(new_valid_cv)    
print('acc = %.4f' % (sum(prediction == valid_labels) / len(valid_labels)))
print(pd.DataFrame(confusion_matrix(valid_labels, prediction)))  

# 新的寻找最相似新闻的代码，取向量的曼哈顿距离最近者 

def Find2(query_str):
    seg = jieba.cut(query_str)
    seg = " ".join(seg).split()
    vec = np.zeros(300)
    for word in seg:
        if word in self_train_model.wv.vocab:   
            vec += self_train_model[word]/len(seg)
        else:
            vec += np.zeros(300)
    score = np.zeros(len(new_train_cv))
    for i in range(len(new_train_cv)): 
        diff = np.array(vec) - np.array(new_train_cv[i])
        score[i] = abs(diff).sum()
    ids = list(range(len(new_train_cv)))    
    ids.sort(key=lambda x:score[x])
    for i in ids[:20]: 
        print(train_raw[i][0], train_raw[i][1])     
    print("\n")   

Find2("王者荣耀国际版入选东南亚运动会电竞项目")  
Find2("首只独角兽周二上市！8个涨停赚4万")     
Find2("名爵EZS上市最低仅需9.99万元 高颜值更靠谱的纯电动SUV")    


################ End
