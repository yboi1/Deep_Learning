import torchvision

import torch
from torch import nn
import torch.nn.functional as F
# LeNet
# 七层： 两次卷积层两层池化层， 最后三层全连接层
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5, padding=1))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # 将数据转换为二维， 保证后续 Linear 的正确调用
        x = x.view(x.size(0), -1)
        x = self.layer3(x)


#     第一层卷积层 (conv1 + pool1):
#         输入数据形状：(batch_size, 1, height, width)
#         nn.Conv2d(1, 6, 3, padding=1)：输入通道为 1，输出通道为 6，卷积核尺寸为 3x3，填充为 1。
#         卷积后的输出形状：(batch_size, 6, height, width)
#         nn.MaxPool2d(2, 2)：2x2 的最大池化层，步幅为 2。
#         池化后的输出形状：(batch_size, 6, height/2, width/2)
#
#     第二层卷积层 (conv2 + pool2):
#         输入数据形状：(batch_size, 6, height/2, width/2)
#         nn.Conv2d(6, 16, 5, padding=1)：输入通道为 6，输出通道为 16，卷积核尺寸为 5x5，填充为 1。
#         卷积后的输出形状：(batch_size, 16, height/2, width/2)
#         nn.MaxPool2d(2, 2)：2x2 的最大池化层，步幅为 2。
#         池化后的输出形状：(batch_size, 16, height/4, width/4)
#
#     展平数据：
#         输入数据形状：(batch_size, 16, height/4, width/4)
#         x.view(batch_size, -1)：将输入展平为形状为 (batch_size, 16 * height/4 * width/4) 的二维张量。
#
#     全连接层 (fc1, fc2, fc3):
#         输入数据形状：(batch_size, 16 * height/4 * width/4)
#         nn.Linear(400, 120)：输入特征维度为 400，输出特征维度为 120。
#         nn.Linear(120, 84)：输入特征维度为 120，输出特征维度为 84。
#         nn.Linear(84, 10)：输入特征维度为 84，输出特征维度为 10。


# =====================================================================
# AlexNet
# 第一次引入激活函数Relu 在全连接层加入Dropout防止过拟合
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),   # 是否在原数据上改变
            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 192, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),  # 是否在原数据上改变
            nn.MaxPool2d(3, 2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), )

        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classfier(0)


# =====================================================================
# VGGNet
# 使用了更小的滤波器，同时使用了更深的结构


# =====================================================================
# GoogleNet
# 采用了22层 但是参数比AlexNet少了12倍  采用了Inception模块，且没有全连接层
# Inception模块   用几个并行的滤波器对输入进行卷积和池化 最后将输出的结果按深度拼接在一起形成输出层

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernal_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernal_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3db1_1 = BasicConv2d(in_channels, 64, kernal_size=1)
        self.branch3x3db1_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3db1_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(
            in_channels, pool_features, kernel_size=1
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3db1 = self.branch3x3db1_1(x)
        branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db1_3(branch3x3db1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3db1, branch_pool]

        return torch.cat(outputs, 1)

#  首先定义一个最基础的卷积模块， 然后定义1x1, 3x3, 5x5的模块和一个池化层，
#  最后使用torch.cat将其拼接起来  得到输出结果


# =====================================================================
# ResNet
# 通过残差模块训练出一个高达152层的神经网络
# 不再学习一个完整的输出， 而是学习输出和输入的差别 H(x) - x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


    torchvision.models.ResNet()