import glob

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

label_name = ["airplane","automobile","bird","cat",
              "deer","dog", "frog",
              "horse", "ship","truck"]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx

def default_loader(path):
    return Image.open(path).convert("RGB")

#   简易的数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(28),
    # 水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

#   较高强度的数据增强

# train_transform = transforms.Compose([
#     # 随机裁剪
#     transforms.RandomResizedCrop((28, 28)),
#     # 水平翻转
#     transforms.RandomHorizontalFlip(),
#     # 垂直翻转
#     transforms.RandomVerticalFlip(),
#     # 翻转
#     transforms.RandomRotation(90),
#     # 灰度图像
#     transforms.RandomGrayscale(0.1),
#     # 颜色，亮度对比度等
#     transforms.ColorJitter(),
#
#     transforms.ToTensor()
# ])

class MyDataset(Dataset):
    def __init__(self, im_list,
                 transform=None,
                 loader = default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        # img_list 是指某一类别图片的地址
        for im_item in im_list:
            im_label_name = im_item.split("/")[-2]
            # 类别及图片名称传入到列表中 路径、ID
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.loader = loader
        self.transform = transform

    # 对数据的读取和对数据的增强， 返回图片的数据和Label
    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob("/home/boyi/Code/Pytorch-program/TRAIN/*/*.png")
im_test_list = glob.glob("/home/boyi/Code/Pytorch-program/TEST/*/*.png")

train_dataset = MyDataset(im_train_list,
                         transform = train_transform)
test_dataset = MyDataset(im_test_list,
                        transform = test_transform)

train_loader = DataLoader(dataset=train_dataset,
                              batch_size=6,
                              shuffle=True,
                              num_workers=4)
test_loader = DataLoader(dataset=train_dataset,
                              batch_size=6,
                              shuffle=False,
                              num_workers=4)

print("num_of_train", len(train_dataset))
print("num_of_test", len(test_dataset))