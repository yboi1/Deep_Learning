import numpy as np
def function_2(x):
    return x[0]**2 + x[1]**2

#其中f为函数, x为一个数组(np.array)
def numerical_gradient(f, x):
    h = 1e-4
    #np.zero_like 用来生成一个与x形状相同的数组
    grad = np.zeros_like(x)
    #遍历x中的数值
    for idx in range(x.size):
        tmp_val = x[idx]
        #
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

a = numerical_gradient(function_2, np.array([3.0, 4.0]))
b = numerical_gradient(function_2, np.array([2.0, 2.0]))
print(a)
print(b)

def gradient_descent(f, init_x, lr=0.01, stem_num=100):
    x = init_x
    
    for i in range(stem_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x