import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

dataLoader = DataLoader(test_set, 64, shuffle=True, num_workers=0)

# 测试集中的数据
img, label = test_set[0]
print(img.shape)
print(label)

writer = SummaryWriter("../dataloader")
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