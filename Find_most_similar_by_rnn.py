# 加载已经训练好的LSTM模型，获取句子向量，找到最相似的句子 

from dataloader import load_data
test_entry = load_data('dataset/test.pkl')
similarity = []

import jieba
import torch
import pickle 
import numpy as np 

from torch.autograd import Variable

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

# 加载模型
model = torch.load('model/w2v_nonstop_512_0.95decay_attn[2].pkl',map_location='cpu')
word2idx = load_data('word2idx.pkl')
label2id = {'news_culture': 0,
            'news_entertainment': 1,
            'news_sports': 2,
            'news_finance': 3,
            'news_house': 4,
            'news_car': 5,
            'news_edu': 6,
            'news_tech': 7,
            'news_military': 8,
            'news_travel': 9,
            'news_world': 10,
            'news_game': 11}
def Word2idx(word):
    try:
        return word2idx[word]
    except:
        return word2idx['Unknown']
        
def Get_sentence_type_and_embedding(sentence):
    word_list = list(jieba.cut(sentence))
    word_index = [Word2idx(word) for word in word_list]
    #print(word_index)
    out1, out2 = model(Variable(torch.LongTensor([word_index])))
    index = torch.squeeze(out1).max(0)[1].item()
    return list(label2id.keys())[index], torch.squeeze(out2).cpu().data.numpy()

# 获取所有句子的编码    
train_cv = [] 
for i,doc in enumerate(train_docs): 
    t,v = Get_sentence_type_and_embedding(doc) 
    train_cv.append(v)  
    if i % 100 == 0 :
        print(i) 
        

def Find_most_similar(sentence):
    type,vect = Get_sentence_type_and_embedding(sentence) 
    score = np.zeros(len(train_docs)) 
    for i,doc in enumerate(train_docs):
        v = train_cv[i]   
        diff = np.array(vect) - np.array(v) 
        score[i] = abs(diff).sum() 
        
    ids = list(range(len(train_cv)))    
    ids.sort(key=lambda x:score[x])
    for i in ids[:20]: 
        print(train_raw[i][0], train_raw[i][1])     
    print("\n")   

# 寻找最相似的句子

Find_most_similar("王者荣耀国际版入选东南亚运动会电竞项目") 
Find_most_similar("首只独角兽周二上市！8个涨停赚4万") 
Find_most_similar("名爵EZS上市最低仅需9.99万元 高颜值更靠谱的纯电动SUV") 