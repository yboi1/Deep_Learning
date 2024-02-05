import numpy as np
from torch import nn
import torch

basic_rnn = nn.RNN(input_size=20, hidden_size=50, num_layers=2)
# print(basic_rnn.weight_ih_l0)       # 获取第一层中的Weight

toy_input = torch.randn(100, 32, 20)
h_0 = torch.randn(2, 32, 50)

toy_output, h_n = basic_rnn(toy_input, h_0)
print(toy_output.size())
print(h_n.size())

lstm = nn.LSTM(input_size=20, hidden_size=50, num_layers=2)
lstm_out, (h_n, c_n) = lstm(toy_input)
print(lstm_out.size())
print(toy_output.size())
print(h_n.size())

class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)

        return out


# 将数据标准化到0~1    建立数据集
def creat_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i: (i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

class Lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1,
                 num_layer=2):
        super(Lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        # 先将数据转化为二维， 进行Linear操作
        x = x.view(s*b, h)
        x = self.layer2(x)
        # 将数据变回原来形式
        x = x.view(s, b, -1)
        return x






