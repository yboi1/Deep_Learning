# Pytorch学习

## 测试样例

net: 模型， num：训练epoch， 

```python
# train
def train_model(net,num,conv=False):
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(num):
        for im,_ in train_data:
            if not conv:
                im = im.view(im.shape[0],-1)
            im = Variable(im)
            _,output = net(im)
            loss = criterion(output, im)/im.shape[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:  # 每 20 次，将生成的图片保存一下
            print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.data[0]))
            pic = to_img(output.cpu().data)
            if not os.path.exists('./simple_autoencoder'):
                os.mkdir('./simple_autoencoder')
            save_image(pic, './simple_autoencoder/image_{}.png'.format(epoch + 1))
```





# 视频课程

```python
#所有的数据集都要引入Dataset并重写__getitem__()函数
from torch.utils.data import Dataset
import os 	#os 是用来读取路径

from PIL import Image

class MyData(Dataset):
	#root_dir 根目录文件名
    #label_dir 工作文件名
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        
        #os.path.join() 函数用于将两个地址相拼接,避免出现错误
        #拼接完后的地址及为图片所在地址
        self.path = os.path.join(root_dir, label_dir)
        
        #os.listdir() 是将文件夹下的所有图片地址存在一个列表中
        self.img_path = os.listdir(self.path)

        #这里是重写父类函数,idx为[]内下标 从0开始
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, 									img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
```

```python
#测试用例
root_dir = "dataset"
Mine_dir = "test"
mine_dataset = MyData(root_dir, Mine_dir)
img, label = mine_dataset[0]
img.show()


```



#### Dataset

> 下载一些现有的数据集
>
> rooot表示下载到的地址, transform表示下载后的格式, train:是否为训练数据

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", 				transform=dataset_transform, train=True, download=True) 
test_set = torchvision.datasets.CIFAR10(root="./dataset", 				transform=dataset_transform, train=True, download=True)


# print(type(train_set))    <class 'torchvision.datasets.cifar.CIFAR10'>
# img, target = train_set[1]
# print(img)
# print(target)
# print(test_set.classes)
# img.show()

# print(test_set[1])

#img为图片,类型为Tensor
writer = SummaryWriter("logs")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("test", img, i)

writer.close()
```



#### DataLoader

> 返回的是一个迭代器
>
> 参数:
>
> dataset 数据集,	 batch_size 一次取几张,	shuffle 是否重新打乱顺序
>
> num_workers 是否采用多线程	drop_last	整取完图片后剩余图片是否使用

```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("./dataset", train=False, 				transform=torchvision.transforms.ToTensor())

dataLoader = DataLoader(test_set, 64, shuffle=True, num_workers=0)

# 测试集中的数据
img, label = test_set[0]
print(img.shape)
print(label)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in dataLoader:
        imgs, label = data
        # print(imgs.shape)
        # print(label)
        # add_images用于读取多个图片
        writer.add_images(f"epoch{epoch}", imgs, step)
        step += 1

writer.close()
```





#### TensorBoard

> 执行完下述代码后,会在logs文件下生成一个对应文件
>
> 在pytorch或者anaconda prompt 下输入tensorboard --logdir=logs
>
> 生成一个连接点击连接即可查看图片

```python
from torch.util.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

#图片的地址
image_path = "dataset/test/DA4E63CDA281F9BE5E9BE5A713FD8AD8.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
#print(type(img_array))
#print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats="HWC")

#x轴为global_step 	y轴为scalor_value
for i in range(-50, 50):
	writer.add_scalar("y=x*3", i*3, i)
    #函数
writer.close()
```



#### Transforms

```python
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms		#导入transforms库
from PIL import Image			#将导入的图片转化为													PIL.JpegImagePlugin.JpegImageFile

img_path = "dataset/test/2022-08-27 15.07.51(1).jpg"
img = Image.open(img_path)

#SummaryWriter类用于作图,记录学习过程
Writer = SummaryWriter("logs")


# transforms 该如何使用:
#	先创建一个实例对象,再调用其函数进行赋值
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 为什么转化为Tensor类型:   内置了深度学习中大多需要的参数

Writer.add_image("Tensor_img", tensor_img)

Writer.close()

# import cv2
# cv_img = cv2.imread(img_path)
```



#### CommClass



##### Compose

```python
# Class -Compose                将各种工具组合
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
])

img_path = "dataset/test/2022-08-27 15.07.51(1).jpg"
img = Image.open(img_path)
img = transform(img)
print(type(img))
```



##### ToTensor

```python
# Class -ToTensor              将图片转化为Tensor类型,便于深度学习的进行
img = cv2.imread(img_path)
tensor = transforms.ToTensor()
img_tensor = tensor(img)
print(type(img_tensor))
```



##### Normalize

```python
# Class -Normalize              正规化,保证数据位于y轴两侧,范围对称
print(img_tensor[2][1][2])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_tensor[2][1][2])
```



##### Resize

> 调整图片的尺寸大小

```python
# Resize
print(img_2.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_2)
img_resize = tensor(img_resize)
print(img_resize)
```



##### RandomCrop

> 随机裁剪图片

```python
# RandomCrop


trans_random = transforms.RandomCrop(24)
trans_compose_2 = transforms.Compose([trans_random, tensor])
for i in range(100):
    img_crop = trans_compose_2(img_3)
    writer.add_image("RandomCrop1", img_crop, i)
```



## 深度学习



### 卷积

```python
import torch
import torch.nn.functional as F

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])

ternel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(ternel, (1, 1, 3, 3))

print(input.shape)
print(ternel.shape)

# kernel_size 卷积核的大小，X*X
# padding 为 填充大小
# conv几d 表示的是几维卷积
# stride 表示路径大小为多少
# bias表示偏置
# padding_mode 表示填充参数
output = F.conv2d(input, kernel, stride=1)
print(output)
```



### 卷积层

```python
class TuDui(nn.Module):
    def __init__(self):
        super(TuDui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x
```



### 池化层

最大池：

```python
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output
```





# 书本

## Tensor

```python
# 展示矩阵的元素和大小
import numpy as np
import torch
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
# print("a is: {}".format(a))
# print('a size is {}'.format(a.size()))

# 改变矩阵的数据类型
b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])  # longInt
# print('b is : {}'.format(b))

# 创建一个全零的空Tensor
c = torch.zeros((3, 2))
# print('zero tensor" {}'.format(c))

# 取一个正态分布为随机初始值
d = torch.randn((3, 2))
# print('d is {}'.format(d))

# Tensor 与 numpt.ndarray相互转换
numpy_b = b.numpy()
# print("conver to numpy is \n {}".format(numpy_b))

e = np.array([[2, 3], [4, 5]])
torch_e = torch.from_numpy(e)
print('from numpy to torch is: {}'.format(torch_e))

f_torch_e = torch_e.float()
print('change data type to float tensor is {}'.format(f_torch_e))


# cuda 加速
print(torch.cuda.is_available())
if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)
```





## Variable

> 已经被淘汰， 自动都会求导

```python
import torch
from torch.autograd import Variable

# Variable 中提供了自动求导的功能  适用于前向后向传播
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

#   在PyTorch的更新版本中（1.0版本及以后），
#   Variable 已经被弃用，而且不再需要显式地使用 Variable 包装张量，
#   因为张量默认就具备了自动求导的功能。
y = w * x + b

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)


# 矩阵求导
x = torch.randn(3)

y = x * 2
print(y)

y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)
```



## 模型的保存和加载

> 保存：save
>
> 加载：load

```python
import torch
torch.save(model, './model.pth')
torch.save(model.state_dict(), './model_state.pth')

load_model = torch.load('model.pth')
model.load_state_dic(torch.load('model_state.pth'))
```



## 单维度神经

```python
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

class LinerRegression(nn.Module):
    def __init__(self):
        super(LinerRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

if torch.cuda.is_available():
    model = LinerRegression().cuda()
else:
    model = LinerRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# num_epochs = 1000
# for epoch in range(num_epochs):
#     if torch.cuda.is_available():
#         inputs = Variable(x_train).cuda()
#         target = Variable(y_train).cuda()
#     else:
#         inputs = Variable(x_train)
#         target = Variable(y_train)

# 修改后代码
num_epochs = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    inputs = x_train.to(device)
    target = y_train.to(device)


    # forward
    out = model(inputs)
    loss = criterion(out, target)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.item()))

model.eval()
predict = model(x_train)
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.show()
```



## 拟合原函数

```python
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1, 4)], 1)
    # 矩阵的拼接


W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    return x.mm(W_target) + b_target[0]

def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x.to(device), y.to(device)

class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    batch_x, batch_y = get_batch()

    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break

print(W_target)
print(b_target)
```



## 简单的网络

```python
from torch import nn
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
print(model)
```



## 提取网络结构

```python
model = SimpleCNN()
# print(model)

new_model = nn.Sequential(*list(model.children())[:4])	# 数字表示提取几层
# print(new_model)
```



## 提取卷积层

> 提取所有的卷积层 ：isinstence

```python
conv_model = nn.Sequential()
index = 1  # A counter to generate unique names
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        unique_name = f"conv{index}"
        conv_model.add_module(unique_name, layer)
        index += 1

print(conv_model)
```



## 权值初始化

```python
# 权值初始化
for m in model.modules():
    if isinstance(m ,nn.Conv2d):
        init.normal(m.weight.data)
        init.xavier_normal(m.weight.data)
        init.kaiming_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_()
```



对于卷积层 (`nn.Conv2d`)：

- `init.normal(m.weight.data)`：这行代码使用均值为0、标准差为1的正态分布随机初始化卷积核的权重（`m.weight.data`）。
- `init.xavier_normal(m.weight.data)`：这行代码使用 Xavier 初始化方法，根据输入和输出的连接数来初始化权重。这种方法有助于在激活函数前后保持梯度的稳定性，使训练更加稳定。
- `init.kaiming_normal(m.weight.data)`：这行代码使用 Kaiming He 初始化方法，也称为 "He Initialization"，针对 ReLU 激活函数的网络设计。它考虑了激活函数的非线性特性，有助于提高训练效果。
- `m.bias.data.fill_(0)`：这行代码将卷积层的偏置（bias）初始化为全零。

对于线性层 (`nn.Linear`)：

- `m.weight.data.normal_()`：这行代码使用均值为0、标准差为1的正态分布随机初始化线性层的权重（`m.weight.data`）。注意，这里使用了下划线（`_`）表示原地修改参数值。



## 循环神经网络

### RNN、LSTM、GRU

```python
basic_rnn = nn.RNN(input_size=20, hidden_size=50, num_layers=2)
# print(basic_rnn.weight_ih_l0)       # 获取第一层中的Weight

toy_input = torch.randn(100, 32, 20)
h_0 = torch.randn(2, 32, 50)

toy_output, h_n = basic_rnn(toy_input, h_0)
print(toy_output.size())
print(h_n.size())

lstm = nn.LSTM(input_size=20, hidden_size=50, num_layers=2)
lstm_out, (h_n, c_n) = lstm(toy_input)
print(lstm_out.size())
print(toy_output.size())
print(h_n.size())
```



### 序列模型

```python

# 将数据标准化到0~1    建立数据集
def creat_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i: (i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

class Lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1,
                 num_layer=2):
        super(Lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        # 先将数据转化为二维， 进行Linear操作
        x = x.view(s*b, h)
        x = self.layer2(x)
        # 将数据变回原来形式
        x = x.view(s, b, -1)
        return x
```





## 自动编码器

```python
class DCautoencoder(nn.Module):
    def __init__(self):
        super(DCautoencoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)            
        )
        
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),		# 反
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
            
        )
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        
        return x
```

