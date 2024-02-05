import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        # 调用父类的构造函数，固定格式
        super(Tudui, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
print(x)
output = tudui(x)
print(output)




