import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([
#     [1, 2, 0, 3, 1],
#     [1, 2, 0, 3, 1],
#     [1, 2, 0, 3, 1],
#     [1, 2, 0, 3, 1],
#     [1, 2, 0, 3, 1]
# ], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("../dataset", False, torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()
writer = SummaryWriter("../log")

step = 0
for data in dataLoader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()


