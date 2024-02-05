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