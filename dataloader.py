import pickle
import jieba
import torch
import torch.utils.data

VOCAB_SIZE = 84316

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

def Read_and_save():
    with open("dataset/train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("dataset/valid.pkl", "rb") as f:
        valid_data = pickle.load(f)

    train_raw, train_docs, train_labels = split_words(train_data)
    valid_raw, valid_docs, valid_labels = split_words(valid_data)
    vocab_set = set()
    for s in train_docs:
        s = s.split(' ')
        for word in s:
            vocab_set.add(word)
    for s in valid_docs:
        s = s.split(' ')
        for word in s:
            vocab_set.add(word)
    print(vocab_set)
    print(len(vocab_set))
    vocab_set.remove('')
    pickle.dump(vocab_set, open('vocab.pkl', 'wb'))

# 加载pkl格式的数据
def load_data(path):
    return pickle.load(open(path, 'rb'))

# 保存成pkl格式的数据
def save_data(obj, path):
    pickle.dump(obj, open(path, 'wb'))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.dataLen = len(data)
        self.data = data
        self.label = label

    def __getitem__(self, index):
        # data[index]: 句子的index表示
        # label[index]: 该句子的label
        return self.data[index], self.label[index], len(self.data[index])

    def __len__(self):
        return self.dataLen

# 对数据进行padding
def collate_fun(data):
    data_label_len = sorted(data, key=lambda x: len(x[0]), reverse=True)
    max_sentence_length = len(data_label_len[0][0])
    data_tensor = []
    for sentence, label, l in data_label_len:
        inp = [sentence[i] if i < len(sentence) else VOCAB_SIZE-1 for i in range(max_sentence_length)]
        data_tensor.append(inp)
    data_tensor = torch.LongTensor(data_tensor)
    target_tensor = torch.LongTensor([i for _, i, __ in data_label_len])
    real_length_tensor = torch.LongTensor([i for _, __, i in data_label_len])
    return data_tensor, target_tensor, real_length_tensor

# 获取dataloader对象
def Get_dataloader(batch_size, filename, shuffle):
    data_and_label = load_data(filename)
    data_train = [i for i, j in data_and_label]
    label_train = [j for i, j in data_and_label]
    # 把 dataset 放入 DataLoader
    trainloader = torch.utils.data.DataLoader(
        dataset=Dataset(data_train, label_train),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fun,
        drop_last=True
    )
    return trainloader
