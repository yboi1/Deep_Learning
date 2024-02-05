#coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
#新增的库文件
#====================
from common.functions import sigmoid, softmax
import pickle

#====================

#获取测试数据集
def get_data():
    
    
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

#读取文件中记录的权重和偏置
def init_network():
    with open(r"C:\Users\Lenovo\Desktop\Program\Deep Learning\Pyrhon-Program\Neural Network\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network

# 验证当前的权重与偏置的正确率


def predict(network, x):
    # 先从文件中读取数据
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 计算 矩阵相乘,带入sigmoid函数,重复
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) +b3
    y = softmax(a3)  #将各数据转化为概率集
    return y


x, t = get_data()
network = init_network()


# 记录正确的个数
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
        
print("Accuracy:" + str(float(accuracy_cnt)/len(x))) #计算正确的百分比
