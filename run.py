import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
from sklearn import preprocessing
import matlab
import matlab.engine
import shutil
import argparse


class Config(object):

    """配置参数"""
    def __init__(self):
        self.save_path = 'model/model.ckpt'   # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                         # 随机失活
        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 103                                           # 类别数
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 2048                                          # mini-batch大小
        self.pad_size = 3                                            # 每句话处理成的长度(短填长切)
        self.lenlen = 72
        self.learning_rate = 5e-4                                       # 学习率
        self.embed = 100
        self.dim_model = 100
        self.hidden = 512
        self.last_hidden = 512
        self.num_head = 10
        self.num_encoder = 8


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Linear(config.lenlen, config.embed)
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])
        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


def test(config, model, x_test):
    model.load_state_dict(torch.load(config.save_path, map_location = torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        texts = torch.tensor(x_test)
        outputs = model(texts)
        m = nn.Sigmoid()
        outputs = m(outputs)
        predic = outputs.data.cpu().numpy()
        predic[predic > 0.5] = 1
        predic[predic <= 0.5] = 0
        pred_id = np.amax(predic, 1)
        predic_append = np.zeros((predic.shape[0], 1))
        for yucezhi in range(0, predic.shape[0]):
            if pred_id[yucezhi] == 0:
                predic_append[yucezhi, 0] = 1
            else:
                predic_append[yucezhi, 0] = 0
        predic = np.hstack((predic, predic_append))
        yuce_label = []
        temp = []
        for hang in range(predic.shape[0]):
            temp = []
            for lie in range(predic.shape[1]):
                if predic[hang, lie] == 1:
                    temp.append(lie)
            if len(temp) > 9:
                print('10 eroor')
            for buchong in range(10 - len(temp)):
                temp.append(104)
            yuce_label.append(temp)
        yuce_label = np.array(yuce_label)
        yuce_label = yuce_label.reshape(-1, 10)
        return yuce_label


def setdir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


if __name__ == '__main__':

    setdir('data/global')
    setdir('data/global_local')
    setdir('data/local')
    setdir('data/result')
    setdir('predict_label')
    parser = argparse.ArgumentParser(description='datapath')
    parser.add_argument('--datapath', type=str, default='data/demo.vtk')
    args = parser.parse_args()
    engine = matlab.engine.start_matlab()
    fame = args.datapath
    print('Calculating FiberGeoMap......')
    print(engine.c_m_c40(fame))
    print(engine.global_local())
    path0 = 'data/global_local/'
    path_list0 = os.listdir(path0)
    for i in range(0, len(path_list0)):
        if path_list0[i][0] == '.':
            path_list0[i] = ''
    while '' in path_list0:
        path_list0.remove('')
    dataset1 = 'data/global_local/' + path_list0[0]
    config = Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    x_test = np.loadtxt(dataset1)
    x_test = np.reshape(x_test, [x_test.shape[0], 3, 72])
    for w in range(0, x_test.shape[0]):
        x_test[w, :, :] = preprocessing.scale(x_test[w, :, :], axis=1)
    x_test = x_test.astype('float32')
    model = Model(config).to(config.device)
    yuce_label = test(config, model, x_test)

    if len(path_list0) > 1:
        for i in range(1, len(path_list0)):
            dataset1 = 'data/global_local/' + path_list0[i]
            config = Config()
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True
            x_test = np.loadtxt(dataset1)
            x_test = np.reshape(x_test, [x_test.shape[0], 3, 72])
            for w in range(0, x_test.shape[0]):
                x_test[w, :, :] = preprocessing.scale(x_test[w, :, :], axis=1)
            x_test = x_test.astype('float32')
            model = Model(config).to(config.device)
            yuce_label_temp = test(config, model, x_test)
            yuce_label = np.vstack((yuce_label, yuce_label_temp))
    np.savetxt('predict_label/yuce_label.txt', yuce_label, fmt="%d")
    print('Done.Writing......')
    print(engine.takeVtk(fame))
