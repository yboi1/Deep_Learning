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