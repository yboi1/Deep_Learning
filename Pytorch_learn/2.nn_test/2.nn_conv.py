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

