from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "dataset/test/2022-08-27 15.07.51(1).jpg"
img = Image.open(img_path)

Writer = SummaryWriter("../logs")


# transforms 该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 为什么转化为Tensor类型:   内置了深度学习中大多需要的参数

Writer.add_image("Tensor_img", tensor_img)

Writer.close()

# import cv2
# cv_img = cv2.imread(img_path)