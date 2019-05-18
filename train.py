import torch
from torch import nn
import numpy as np
from model import Train_model
from model import RNN
from model import RNN_with_attention
from dataloader import VOCAB_SIZE
from dataloader import Get_dataloader

# 用不用GPU
Use_CUDA = True

# 超参
EPOCH = 3
BATCH_SIZE = 64
INPUT_SIZE = 768
HIDDEN_SIZE = 512
OUTPUT_SIZE = 12
NUM_LAYERS = 2
LR = 0.001

# train和test文件路径
train_index_data_path = 'train_index_nonstop.pkl'
test_index_data_path = 'valid_index_nonstop.pkl'

# 数据和embedding权重的加载
train_loader = Get_dataloader(BATCH_SIZE, train_index_data_path, True)
valid_loader = Get_dataloader(1000, test_index_data_path, False)
weight = np.load('weight_matrix_bert.npy')

# 模型
model = RNN_with_attention(
    vocab_size=VOCAB_SIZE,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_layers=NUM_LAYERS,
    pretrained_weight=weight
)
if Use_CUDA: model = model.cuda()

if __name__ == '__main__':
    Train_model(
        model = model,
        train_loader = train_loader,
        test_loader = valid_loader,
        EPOCH = EPOCH,
        optimizer = torch.optim.Adam(model.parameters(), lr=LR),
        init_LR = LR, # 初始的学习率
        loss_func = nn.CrossEntropyLoss(ignore_index=VOCAB_SIZE-1),
        Use_CUDA = Use_CUDA,
        save_path = 'model/bert_nonstop_512_0.95decay_sota' # 不用加.pkl
    )
