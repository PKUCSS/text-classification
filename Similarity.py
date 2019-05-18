from dataloader import load_data
from pyhanlp import HanLP
import jieba
import torch
from torch.autograd import Variable
def Get_dependency(sentence):
    return HanLP.parseDependency(sentence)

model = torch.load('model/w2v_nonstop_128_0.95decay_attn[1].pkl')
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

# 返回一个句子的类型和经过神经网络训练后的embedding
def Get_sentence_type_and_embedding(sentence):
    word_list = list(jieba.cut(sentence))
    word_index = [Word2idx(word) for word in word_list]
    out1, out2 = model(Variable(torch.LongTensor([word_index])).cuda())
    index = torch.squeeze(out1).max(0)[1].item()
    return list(label2id.keys())[index], torch.squeeze(out2).cpu().data.numpy()

# 输出所有预测错误的句子
def Output_all_errors():
    data, label = load_data('valid_docs_new.pkl')
    cnt = 0
    for d, l in zip(data, label):
        c = Get_sentence_type_and_embedding(''.join(d))[0]
        if label2id[c] != l:
            print(''.join(d), 'predict:' + str(c), 'truth:'+str(list(label2id.keys())[l]))
            cnt += 1
    print(cnt)

# 每次输入一个句子，预测
def Try():
    while(1):
        s = input()
        dp = Get_dependency(''.join(s))
        print('Topic:', Get_sentence_type_and_embedding(s)[0])
        print(dp)
