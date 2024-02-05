from torch import nn
from torch.nn import init


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()       # 3*32*32
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1))   # 32*32*32
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))      # 32, 16, 16
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1))  # 64, 16, 16
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))  # 32, 8, 8
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))  # 128, 8, 8
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))  # 128, 4, 4
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer2(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

model = SimpleCNN()
# print(model)

new_model = nn.Sequential(*list(model.children())[:4])
# print(new_model)


# 错误
# conv_model = nn.Sequential()
# for layer in model.named_modules():
#     if isinstance(layer[1], nn.Conv2d):
#         conv_model.add_module(layer[0], layer[1])
#
# print(conv_model)

# 提取所有的卷积层
conv_model = nn.Sequential()
index = 1  # A counter to generate unique names
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        unique_name = f"conv{index}"
        conv_model.add_module(unique_name, layer)
        index += 1

# print(conv_model)

# 提取参数及自定义初始化
for params in model.named_parameters():
    print(params[0])
    # params[1] 为Tensor类型的数据

# 权值初始化
for m in model.modules():
    if isinstance(m ,nn.Conv2d):
        init.normal(m.weight.data)
        init.xavier_normal(m.weight.data)
        init.kaiming_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_()

#     对于卷积层 (nn.Conv2d)：
#
#         init.normal(m.weight.data)：这行代码使用均值为0、标准差为1的正态分布随机初始化卷积核的权重（m.weight.data）。
#
#         init.xavier_normal(m.weight.data)：这行代码使用 Xavier 初始化方法，根据输入和输出的连接数来初始化权重。这种方法有助于在激活函数前后保持梯度的稳定性，使训练更加稳定。
#
#         init.kaiming_normal(m.weight.data)：这行代码使用 Kaiming He 初始化方法，也称为 "He Initialization"，针对 ReLU 激活函数的网络设计。它考虑了激活函数的非线性特性，有助于提高训练效果。
#
#         m.bias.data.fill_(0)：这行代码将卷积层的偏置（bias）初始化为全零。
#
#     对于线性层 (nn.Linear)：
#         m.weight.data.normal_()：这行代码使用均值为0、标准差为1的正态分布随机初始化线性层的权重（m.weight.data）。注意，这里使用了下划线（_）表示原地修改参数值。