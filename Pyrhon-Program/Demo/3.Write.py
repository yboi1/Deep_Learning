from two_layer import TwoLayerNet
#第一步,数据准备
import numpy as np
mnist = np.load(r"C:\Users\Lenovo\Desktop\Program\Deep Learning\Pyrhon-Program\Demo\mnist.npz")

x_train = mnist['x_train']
y_train = mnist['y_train']

x_test = mnist['x_test']
y_test = mnist['y_test']

#将图像转化为一维数据
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

#第二步 搭建神经网络
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#计算概率分布
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
    
#三层模型
# input_size = 784 #一张图片的像素点 28*28
# hidden_size = 128
# output_size = 10

#权重初始化
# W1 = np.random.randn(input_size, hidden_size)
# b1 = np.zeros(hidden_size)
# W2 = np.random.randn(hidden_size, output_size)
# b2 = np.zeros(output_size)

#上述两步可替换为类成员
network = TwoLayerNet(28*28, 128, 10)

#第三步 模型训练

train_loss_list = []  # 初始化训练损失列表
train_acc_list = []   # 初始化训练准确率列表
test_acc_list = []    # 初始化测试准确率列表

#测试数据个数,每个batch的数目,学习速率
learning_rate = 0.1
epochs = 5
batch_size = 100
train_size = x_train.shape[0]

iter_per_epoch = max(train_size // batch_size, 1)


for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x = x_train[i:i+batch_size]
        y = y_train[i:i+batch_size]

        batch_mask = np.random.choice(train_size, batch_size)    
        x_batch = x_train[batch_mask]
        t_batch = y_train[batch_mask]

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
            train_acc = network.accuracy(x_train, y_train)
            test_acc = network.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc |" + str(train_acc) + "," +  str(test_acc))