import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform, train=False, download=True)
# print(type(train_set))    <class 'torchvision.datasets.cifar.CIFAR10'>
# img, target = train_set[1]
# print(img)
# print(target)
# print(test_set.classes)
# img.show()

# print(test_set[1])

writer = SummaryWriter("../logs")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("test", img, i)

writer.close()