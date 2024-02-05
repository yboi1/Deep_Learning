
import torch
import torch.nn as nn
import torchvision

from vggnet import VGGNet
from load_cifar10 import train_loader, test_loader
import os

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 200
lr = 0.01
batch_size = 128

net = VGGNet().to(device)

#loss  交叉熵
loss_func = nn.CrossEntropyLoss()

#optimizer  优化器
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
#optimizer = torch.optim.SGD(net.parameters(), lr = lr,
#                              momentum=0.9, weight_decay=5e-4)

# 学习率调整     每进行五次opoch， 学习率变为之前的0.9倍
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=5, gamma=0.9)

for epoch in range(epoch_num):
    print("epoch is ", epoch)


    for i, data in enumerate(train_loader):
        net.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        # 梯度设为零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("step", i, "loss is", loss.item())

        # 在第一个维度上的最大值
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).cpu().sum()

        # print("epoch is ", epoch)

        print("step", i, "loss is", loss.item(),
              "mini-batch correct is:", 100.0 * correct / 128)
        # print("lr is ", optimizer.state_dict()['param_groups'][0]['lr'])

    # if not os.path.exists("models"):
    #     os.mkdir("models")
    # torch.save(net.state_dict(), "models/{}.pth".format(epoch + 1))
    # scheduler.step()
    #
    # print("lr is ", optimizer.state_dict()['param_groups'][0]['lr'])

    # sum_loss = 0
    # sum_correct = 0
    # for i, data in enumerate(test_loader):
    #     net.eval()
    #
    #     inputs, labels = data
    #     inputs, labels = inputs.to(device), labels.to(device)
    #
    #     outputs = net(inputs)
    #     loss = loss_func(outputs, labels)
    #
    #     # 在第一个维度上的最大值
    #     _, pred = torch.max(outputs.data, dim=1)
    #     correct = pred.eq(labels.data).cpu().sum()
    #
    #
    #     sum_loss += loss.items()
    #     sum_correct = correct.item()
    #
    # test_loss = sum_loss * 1,0 / len(test_loader)
    # test_correct = sum_correct * 100.0 / len(test_loader) / batch_size
    #
    # print("epoch is ", epoch+1, "loss is", test_loss,
    #           "test correct is:",test_correct)

