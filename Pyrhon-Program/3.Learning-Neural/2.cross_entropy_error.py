#y表示各输出值的概率
#t代表正确值标签
#此函数最终只会利用正确值的概率,忽略了其他数据
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

#训练数据中随机抽取十个数字
#axis 是一个用于指定沿着哪个轴进行操作的参数。
#在 NumPy 中，ndim 是一个属性，用于获取数组的维度（即数组的轴数）.
#对于一个数组，它可能是一维、二维、三维，或者更高维度的张量。

#改良版,可以处理单个或多个数据 (one-hot形式)
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0] #这里的0表示的是第0维度
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    #当监督数据为标签形式时
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size