import torch
from torch import nn
import torch.cuda
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, num_layers, pretrained_weight=None):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.embedding = nn.Embedding(vocab_size, input_size)
        if pretrained_weight is not None:
            pretrained_weight = np.array(pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 将词语的下标转换成word embedding
        e_x = self.embedding(x)

        # 初始hidden_state为None
        r_out, (h_n, h_c) = self.rnn(e_x, None)

        # 取最后时刻的hidden_state再通过一个全连接层输出
        # 这里不用加softmax, pytorch的CrossEntropyLoss函数已经封装好了log-softmax
        out = self.out(r_out[:, -1, :])
        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(True),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class RNN_with_attention(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, num_layers, pretrained_weight=None):
        super(RNN_with_attention, self).__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = SelfAttention(hidden_size)
        self.embedding = nn.Embedding(vocab_size, input_size)
        if pretrained_weight is not None:
            pretrained_weight = np.array(pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size = x.size(0)
        # 将词语的下标转换成word embedding
        e_x = self.embedding(x)

        # 初始hidden_state为None
        r_out, (h_n, h_c) = self.rnn(e_x, None)
        embedding, attn_weights = self.attention(r_out)
        # 取最后时刻的hidden_state再通过一个全连接层输出
        # 这里不用加softmax, pytorch的CrossEntropyLoss函数已经封装好了log-softmax
        out = self.out(embedding.view(batch_size, -1))
        return out, embedding

class CLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, num_layers,kernel_num,kernel_sizes, pretrained_weight=None):
        super(CLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.conv = nn.ModuleList([nn.Conv2d(1, kernel_num, (K,input_size)) for K in kernel_sizes]) 
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.embedding = nn.Embedding(vocab_size, input_size)
        if pretrained_weight is not None:
            pretrained_weight = np.array(pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # 将词语的下标转换成word embedding
        x = self.embedding(x)
        #print(x.shape)
        x = x.unsqueeze(1) 
        #print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        #print(x.shape) 
        x = x.reshape(-1,3,300) 
        # 初始hidden_state为None
        r_out, (h_n, h_c) = self.rnn(x, None)

        # 取最后时刻的hidden_state再通过一个全连接层输出
        # 这里不用加softmax, pytorch的CrossEntropyLoss函数已经封装好了log-softmax
        out = self.out(r_out[:, -1, :])
        return out
        
class CLSTM_with_attention(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, num_layers,kernel_num,kernel_sizes, pretrained_weight=None):
        super(CLSTM_with_attention, self).__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.conv = nn.ModuleList([nn.Conv2d(1, kernel_num, (K,input_size)) for K in kernel_sizes]) 
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = SelfAttention(hidden_size)
        self.embedding = nn.Embedding(vocab_size, input_size)
        if pretrained_weight is not None:
            pretrained_weight = np.array(pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size = x.size(0)
        # 将词语的下标转换成word embedding
        x = self.embedding(x)
        #print(x.shape)
        x = x.unsqueeze(1) 
        #print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        #print(x.shape) 
        x = x.reshape(-1,3,300) 
        # 初始hidden_state为None
        # 初始hidden_state为None
        r_out, (h_n, h_c) = self.rnn(x, None)
        embedding, attn_weights = self.attention(r_out)
        # 取最后时刻的hidden_state再通过一个全连接层输出
        # 这里不用加softmax, pytorch的CrossEntropyLoss函数已经封装好了log-softmax
        out = self.out(embedding.view(batch_size, -1))
        return out, embedding
                

def Train_model(model, train_loader, test_loader, EPOCH, optimizer, init_LR, loss_func, Use_CUDA, save_path):
    best_acc = 0
    # decay
    for epoch in range(EPOCH):
        if epoch != 0:
            LR = init_LR * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
        for step, (x, y, z) in enumerate(train_loader):
            b_x = Variable(x) # x.shape = (batch_size * max_sentence_length)
            b_y = Variable(y)
            if Use_CUDA:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            model.train()
            output, _ = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                accuracy = Test_model(model, test_loader, Use_CUDA)
                print('Epoch: ', epoch, ' | train loss: %.4f' % loss.data.cpu().numpy(),
                      ' | test accuracy: %.6f' % accuracy)
                if accuracy > best_acc:
                    best_acc = accuracy
                    if best_acc > 0.85:
                        torch.save(model, save_path)
        #torch.save(model, save_path+'[{}].pkl'.format(epoch))

def Test_model(model, test_loader, Use_CUDA):
    auc = 0
    tot = 0
    model.eval()
    for test_x, test_y, l in test_loader:
        if Use_CUDA: test_x = test_x.cuda()
        test_output, _ = model(test_x)
        pred_y = np.squeeze(torch.max(test_output, 1)[1].data.cpu().numpy())
        test_y = test_y.numpy()
        auc += float((pred_y == test_y).astype(int).sum())
        tot += float(test_y.size)
    accuracy = float(auc) / float(tot)
    return accuracy
