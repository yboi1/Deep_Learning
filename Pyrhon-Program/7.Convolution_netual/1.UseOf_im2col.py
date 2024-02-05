import sys, os
sys.path.append(os.pardir)
from common.util import im2col

import numpy as np

#  Parameters
#     ----------
#     input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
#     filter_h : 滤波器的高
#     filter_w : 滤波器的长
#     stride : 步幅
#     pad : 填充

#     Returns
#     -------
#     col : 2维数组
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)