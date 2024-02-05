import numpy as np
mnist = np.load(r"C:\Users\Lenovo\Desktop\Program\Deep Learning\Pyrhon-Program\Demo\mnist.npz")

train_images = mnist['x_train']
train_labels = mnist['y_train']

test_images = mnist['x_test']
test_labels = mnist['y_test']

print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

#将数据转化为一维向量,并进行归一化处理







