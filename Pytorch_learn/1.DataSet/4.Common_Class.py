from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

# Class -Compose                将各种工具组合
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
])

img_path = "dataset/test/2022-08-27 15.07.51(1).jpg"
img = Image.open(img_path)
img_2 = img
img_3 = img
img = transform(img)
print(type(img))

# Class -ToTensor              将图片转化为Tensor类型,便于深度学习的进行
img = cv2.imread(img_path)
tensor = transforms.ToTensor()
img_tensor = tensor(img)
print(type(img_tensor))

# Class -ToPILImage             将Tensor类型的图片转化为PIL格式,用于保存
to_PIL = transforms.ToPILImage()
img_PIL = to_PIL(img_tensor)
print(type(img_PIL))

# Class -Normalize              正规化,保证数据位于y轴两侧,范围对称
print(img_tensor[2][1][2])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_tensor[2][1][2])

# Resize
print(img_2.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_2)
img_resize = tensor(img_resize)
print(img_resize)


# Tensorboard
writer = SummaryWriter("../logs")
writer.add_image("img1", img_resize)
writer.close()

# RandomCrop
trans_random = transforms.RandomCrop(24)
trans_compose_2 = transforms.Compose([trans_random, tensor])
for i in range(100):
    img_crop = trans_compose_2(img_3)
    writer.add_image("RandomCrop1", img_crop, i)




