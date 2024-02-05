### OpenMP 数据管理

#### 全局管理

declare target	声明全局变量 

target update	控制全局变量在主机和设备之间的映射

runtime 函数

#### 局部管理

![image-20230728111607294](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230728111607294.png)



### map子句

> map(映射类型:参数1,参数2)
> 	to 主机端到设备端	from设备端到主机端	tofrom相互映射
>
> ```fortran
> !$omp target map(to:A) map(from:B) map(tofrom:C)
> ```
>
> 

### 数据结构

#### 1>target data

```
!&omp target data map(tofrom:a) map(to:b)	全局使用,避免交互
```

#### 2> target update

不可同时使用to和from子句

```
!$omp target update

#pragma omp target update
```

