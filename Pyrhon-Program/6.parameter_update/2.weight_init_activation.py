import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1] 

    # 出现了梯度消失的情况
    # w = np.random.randn(node_num, node_num) * 1

    # 使用不同数值的标准差进行高斯分布，该条件出现了表现力受限的特点
    # w = np.random.randn(node_num, node_num) * 0.01

    #Xavier初始化，与前一层有n个节点连接时，初始值使用标准差为1/sqrt(n) 的分布
    w = np.random.randn(node_num, node_num) /np.sqrt(node_num)

    z = np.dot(x, w)
    a = sigmoid(z)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()