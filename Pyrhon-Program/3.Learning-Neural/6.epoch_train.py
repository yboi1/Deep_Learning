import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#导入数据
(x_train, t_train), (x_test, t_test) =\
         load_mnist(normalize=True, one_hot_label=True)
         

#测试数据个数,每个batch的数目,学习速率
iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


train_loss_list = []    #损失值
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)


network = TwoLayerNet(784, 50, 10)

for i in range(iters_num):
    
    #随机取样
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)           #高速版
    
    #梯度下降法
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]
        
    #计算损失值
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    #if语句判断一个epoch是否结束
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc |" + str(train_acc) + "," +  str(test_acc))
        
