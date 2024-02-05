import numpy as np
# 抑制过拟合
class Dropout:
    def __init__(self, dropout_radio=0.5):
        self.dropout_radio = dropout_radio
        self.mask = None    # 保存要删除的神经元

    def forward(self, x, train_flg=True):
        if train_flg:
            # 先随机生成和x形状相同的数组，将满足条件的设为True
            self.mask = np.random.randn(*x.shape) > self.dropout_radio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_radio)
        
    # 与ReLu函数的机制相同
    def backward(self, dout):
        return dout * self.mask