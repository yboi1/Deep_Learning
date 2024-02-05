from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs")
image_path = "dataset/test/DA4E63CDA281F9BE5E9BE5A713FD8AD8.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats="HWC")

# global_step 为x轴   scalar_value为y轴
for i in range(-50, 50):
    writer.add_scalar("y=x**3", i**3, i)


writer.close()