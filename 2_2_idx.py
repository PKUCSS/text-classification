import pickle
import jieba
import numpy as np
from nltk.corpus import stopwords 
from sklearn.naive_bayes import GaussianNB 


with open("dataset/train.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("dataset/valid.pkl", "rb") as f:
    valid_data = pickle.load(f)

stopwords = open("stopwords.txt", "r",encoding="utf-8").readlines()
stopwords = [i.strip('\n') for i in stopwords]


word2id = {}
label2id = {}
def split_words(dataset):
    raw_docs = []
    docs = []
    labels = []
    titles = [] 
    global max_len
    for topic, datas in dataset.items():
        if not topic in label2id.keys():
            label2id[topic] = len(list(label2id))
        for data in datas:
            seg = jieba.cut(data["title"])
            raw_docs.append([topic, data["title"]])
           
            cur = []
            for word in seg:
                if word in stopwords: continue
                if not word in word2id:
                    word2id[word] = len(list(word2id))
                cur.append(word2id[word])
            max_len = max(max_len, len(cur))
            titles.append(cur)
            
            docs.append(" ".join(seg))
            labels.append(label2id[topic])
    return raw_docs, docs, labels, titles

max_len = 0
train_raw, train_docs, train_labels, train_titles = split_words(train_data)
valid_raw, valid_docs, valid_labels, valid_titles = split_words(valid_data)

#计算序列编辑距离
def levenshtein_dis(x, y):
    len_str1 = len(x) + 1
    len_str2 = len(y) + 1
    #create matrix
    matrix = [0 for n in range(len_str1 * len_str2)]
    #init x axis
    for i in range(len_str1):
        matrix[i] = i
    #init y axis
    for j in range(0, len(matrix), len_str1):
        matrix[j] = j 
          
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if x[i-1] == y[j-1]:
                cost = 0
            else:
                cost = 2
            matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,
                                        matrix[j*len_str1+(i-1)]+1,
                                        matrix[(j-1)*len_str1+(i-1)] + cost)
    return matrix[-1] / float(len_str1+len_str2-2)
	


N = len(train_titles)
M = len(list(label2id))
score = np.zeros(N)
prediction = []
for title in valid_titles[:1000]:
    for i in range(N):
        score[i] = levenshtein_dis(train_titles[i], title)
    ids = list(range(N))
    ids.sort(key=lambda x:score[x])
    tot = np.zeros(M)
    for i in ids[:20]:
        tot[ train_labels[ids[i]] ] += 1
    prediction.append(np.argmax(tot))
tot = 0
for x in prediction:
    if (x==0): tot += 1
print(tot)



def Find2(query_str):
    seg = jieba.cut(query_str)
    qry = []
    for word in seg:
        if not word in word2id:
            word2id[word] = len(list(word2id))
        qry.append(word2id[word])
        
    score = np.zeros(N)
    for i in range(N):
        score[i] = levenshtein_dis(train_titles[i], qry)
    ids = list(range(N))
    ids.sort(key=lambda x:score[x])
    for i in ids[:20]:
        print(train_raw[i][0], train_raw[i][1])
    print("\n")

Find2("王者荣耀国际版入选东南亚运动会电竞项目")
Find2("首只独角兽周二上市！8个涨停赚4万")
Find2("名爵EZS上市最低仅需9.99万元 高颜值更靠谱的纯电动SUV")