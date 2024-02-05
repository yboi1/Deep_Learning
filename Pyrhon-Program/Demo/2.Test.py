import numpy as np
import matplotlib
import glob
import matplotlib.pyplot as plt

# 步骤1：准备数据
mnist = np.load(r"/home/boyi/Code/Deep_Learning/Pyrhon-Program/Demo/mnist.npz")
train_images, train_labels = mnist['x_train'], mnist['y_train']
test_images, test_labels = mnist['x_test'], mnist['y_test']

# 将图像数据转换为一维向量，并进行归一化处理
train_images = train_images.reshape(-1, 28*28) / 255.0
test_images = test_images.reshape(-1, 28*28) / 255.0

# 步骤2：构建神经网络模型
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

input_size = 28*28
hidden_size = 128
output_size = 10

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)

# 步骤3：模型训练
learning_rate = 0.1
epochs = 5
batch_size = 100

for epoch in range(epochs):
    for i in range(0, len(train_images), batch_size):
        # 前向传播
        X = train_images[i:i+batch_size]
        y = train_labels[i:i+batch_size]
        
        hidden = sigmoid(np.dot(X, W1) + b1)
        output = softmax(np.dot(hidden, W2) + b2)
        
        # 计算损失函数（交叉熵损失）
        loss = -np.sum(np.log(output[np.arange(len(y)), y])) / batch_size
        
        # 反向传播
        d_output = output
        d_output[np.arange(len(y)), y] -= 1
        d_output /= batch_size
        
        dW2 = np.dot(hidden.T, d_output)
        db2 = np.sum(d_output, axis=0)
        
        d_hidden = np.dot(d_output, W2.T) * (hidden * (1 - hidden))
        
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0)
        
        # 参数更新
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
    print(f"Epoch {epoch+1}, Loss: {loss}")

# 步骤4：模型评估
hidden = sigmoid(np.dot(test_images, W1) + b1)
output = softmax(np.dot(hidden, W2) + b2)
predicted_labels = np.argmax(output, axis=1)
accuracy = np.mean(predicted_labels == test_labels)
print("Test accuracy:", accuracy)

num_samples_to_display = 10
indices_to_display = np.random.choice(len(test_images), num_samples_to_display)

# 绘制图像及其预测标签
plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices_to_display):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[idx]}, True: {test_labels[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# #测试模型

# # our own image test data set
# our_own_dataset = []
 
# # load the png image data as test data set
# for image_file_name in glob.glob('my_own_images/tmp5x6k5ry3(1).png'):
#     # use the filename to set the correct label
#     label = int(image_file_name[-5:-4])
 
#     # load image data from png files into an array
#     print("loading ... ", image_file_name)
#     img_array = imageio.imread(image_file_name, as_gray=True)
 
#     # reshape from 28x28 to list of 784 values, invert values
#     img_data = 255.0 - img_array.reshape(784)
 
#     # then scale data to range from 0.01 to 1.0
#     img_data = (img_data / 255.0 * 0.99) + 0.01
#     print(numpy.min(img_data))
#     print(numpy.max(img_data))
 
#     # append label and image data  to test data set
#     record = numpy.append(label, img_data)
#     our_own_dataset.append(record)
 
#     pass
 
# # test the neural network with our own images
 
# # record to test
# item = 3
 
# # plot image
# matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28, 28), cmap='Greys', interpolation='None')
# matplotlib.pyplot.show()
# # correct answer is first value
# correct_label = our_own_dataset[item][0]
# # data is remaining values
# inputs = our_own_dataset[item][1:]
 
# # query the network
# outputs = n.query(inputs)
# print(outputs)
 
# # the index of the highest value corresponds to the label
# label = numpy.argmax(outputs)
# print("network says ", label)
# # append correct or incorrect to list
# if (label == correct_label):
#     print("match!")
# else:
#     print("no match!")
#     pass