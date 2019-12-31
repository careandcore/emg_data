# TensorFlow

## 1 TensorFlow-keras 简介

### 1.1_keras是什么

* 基于python的高级神经网络API
* Francois Chollet于2014-2015年编写keras
* 以TensorFlow、CNTK或者Theano为后端运行，keras必须有后端才可以运行
  * 后端可以切换，现在多用TensorFlow 

- 极方便于快速实验，帮助用户以最少的时间验证自己的想法

### 1.2_TensorFlow-keras是什么

* TensorFlow对keras API的规范的实现
* 相对于以TensorFlow为后端的keras，TensorFlow-keras与TensorFlow结合更加紧密
* 实现在tf.keras空间下

### 1.3_Tf.keras和keras联系

* 基于同一套API
  * keras程序可以通过改导入方式轻松转为tf.keras程序
  * 反之可能不成立，因为tf.keras有自己的特性

### 1.4_Tf.keras和keras区别

- Tf.keras全面支持eager mode 
  - 只是用keras.Sequential和keras.Model时没有影响
  - 自定义Model内部运算逻辑的时候会有影响 
    - Tf底层API可以使用keras的model.fit等抽象
    - 适用于研究人员

- Tf.keras支持基于tf.data的模型训练
- Tf.keras支持TPU训练 
- Tf.keras支持tf.distribution中的分布式策略
- 其他特性
  - tf.keras可以与TensorFlow中的estimator集成
  - tf.keras可以保存为SaveModel

###  1.5_如何选择

- 如果想用tf.keras的任何一个特性，那么选tf.keras
- 如果后端的互换性很重要，那么选择keras

## 2 分类问题与回归问题

- 分类问题预测的是类别，模型的输出是概率分布
  
- 三分类问题输出例子：[0.2 ,0.7 ,0.1] 
  
- 回归问题预测的是值，模型的输出是是一个实数值

- 参数是逐步调整的 

- 目标函数可以帮助衡量模型的好坏
  - Model A：[0.1, 0.4, 0.5]
  - Model B:   [0.1, 0.2, 0.7]
    - 假设正确结果是2，显然模型a和模型b是错的，用准确率来看，模型a和b没有区别，但实际上模型a比b更接近正确结果。

- **分类问题**

  - 需要衡量目标类别与当前预测的差距

    - 三分类问题输出例子：[0.2, 0.7, 0.1]
    - 三分类真实类别：2->one-hot->[0, 0, 1]

  - one-hot编码，把正整数变为向量表达

    - 生成一个长度不小于正整数的向量，只有正整数的位置处为1，其余位置都为0

  - 平方差损失 

    ![](J:\笔记\tf图片\平方差损失.png)
    
    - 平方差损失举例：
      - 预测值：[0.2，0.7， 0.1]
      - 真实值：[0， 0， 1]
      - 损失函数值：[（0-0）x2+（0.7-0）x2+（0.1-1）x2]x0.5=0.65
    
  - 交叉熵损失：

    ![](J:\笔记\tf图片\交叉熵损失.png)

- **回归问题**
  - 预测值与真实值的差距
  - 平方差损失
  - 绝对值损失

- <font color=blueyello>模型的训练就是调整参数，使得目标函数逐渐变小的过程</font>

### 2.1_实战-模型构建

- Keras搭建分类模型

- Keras回调函数

- Keras搭建回归模型

  - 代码：运行与jupyter notebook

  - ```python
    # 导入各种包
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    %matplotlib inline
    import numpy as np
    import sklearn
    import pandas as pd 
    import os
    import sys
    import time 
    import tensorflow as tf
    from tensorflow import keras
    
    # 打印各种包的信息
    print(tf.__version__)
    print(sys.version_info)
    for module in mpl,np,pd,sklearn,tf,keras:
        print(module.__name__,module.__version__)
        
    #载入数据集
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
    x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
    y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
    print(x_valid.shape, y_valid.shape)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    #定义单个图像显示函数
    def show_single_img(ima_arr):
        plt.imshow(ima_arr, cmap='binary')
        plt.show()
    show_single_img(x_train[2])
    
    #定义多个图像显示函数
    def show_imgs(n_rows, n_cols, x_data, y_data, class_name):
        assert len(x_data) == len(y_data)
        assert n_rows * n_cols < len(x_data)
        plt.figure(figsize= (n_cols * 1.4, n_rows * 1.6))
        for row in range(n_rows):
            for col in range(n_cols):
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index+1)
                plt.imshow(x_data[index], cmap = 'binary',
                           interpolation= 'nearest')
                plt.axis('off')
                plt.title(class_names[y_data[index]])
        plt.show()
    class_names = ['T-shirt', 'Trouser', 'Pillover', 'Dress', 'Coat', 
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    show_imgs(3,5,x_train,y_train,class_names)
    
    # 创建模型
    # tf.keras.models.Sequential
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape = [28, 28]))# 将28*28转换为一维向量
    model.add(keras.layers.Dense(300, activation = 'relu'))# 全连接层01
    model.add(keras.layers.Dense(100, activation = 'relu'))# 全连接层02
    model.add(keras.layers.Dense(10 , activation =  'softmax'))# 全连接层
    """
    # 一步到位
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = [28,28]),
        keras.layers.Dense(300, activation = 'relu'),
        keras.layers.Dense(100, activation = 'relu'),
        keras.layers.Dense(10 , activation =  'softmax')
    ])
    # relu ： y = max(0,x)
    # softmax: 将向量变成概率分布， x= [x1, x2, x3]
    #                               y= [e^x1/sum ,e^x2/sum, e^x3/sum]
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer= 'adam', 
                 metrics = ['accuracy'])
    # 模型一些显示方法
    model.layers
    model.summary()
    
    #训练模型
    # [None, 784] * W + b -> {None, 300} -->W.shape [784, 300], b = [300] ->235500
    history = model.fit(x_train, y_train, epochs = 10, 
             validation_data = (x_valid, y_valid))
    # 查看训练参数变化
    type(history)
    history.history
    
    # 图像展示参数变化
    def plot_learning_curves(history):
        pd.DataFrame(history.history).plot(figsize = (8,5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
    plot_learning_curves(history)
    ```

### 2.2_实战-数据归一化

- ```python
  # 导入各种包
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  %matplotlib inline
  import numpy as np
  import sklearn
  import pandas as pd 
  import os
  import sys
  import time 
  import tensorflow as tf
  from tensorflow import keras
  
  # 打印各种包的信息
  print(tf.__version__)
  print(sys.version_info)
  for module in mpl,np,pd,sklearn,tf,keras:
      print(module.__name__,module.__version__)
      
  #载入数据集
  fashion_mnist = keras.datasets.fashion_mnist
  (x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
  x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
  y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
  print(x_valid.shape, y_valid.shape)
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)
  
  # 归一化 x = (x - u)/ std  u->均值 std->方差
  from sklearn.preprocessing import StandardScaler
  
  scaler = StandardScaler()
  # x_train:[None, 28, 28] -> [None, 784]
  x_train_scaled = scaler.fit_transform(
      x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
  x_valid_scaled = scaler.transform(
      x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
  x_test_scaled = scaler.transform(
      x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
  
  # 创建模型
  # tf.keras.models.Sequential
  """
  model = keras.models.Sequential()
  model.add(keras.layers.Flatten(input_shape = [28, 28]))# 将28*28转换为一维向量
  model.add(keras.layers.Dense(300, activation = 'relu'))# 全连接层01
  model.add(keras.layers.Dense(100, activation = 'relu'))# 全连接层02
  model.add(keras.layers.Dense(10 , activation =  'softmax'))# 全连接层
  """
  # 一步到位
  model = keras.models.Sequential([
      keras.layers.Flatten(input_shape = [28,28]),
      keras.layers.Dense(300, activation = 'relu'),
      keras.layers.Dense(100, activation = 'relu'),
      keras.layers.Dense(10 , activation =  'softmax')
  ])
  # relu ： y = max(0,x)
  # softmax: 将向量变成概率分布， x= [x1, x2, x3]
  #                               y= [e^x1/sum ,e^x2/sum, e^x3/sum]
  model.compile(loss = 'sparse_categorical_crossentropy', optimizer= 'adam', 
               metrics = ['accuracy'])
  
  #训练模型
  # [None, 784] * W + b -> {None, 300} -->W.shape [784, 300], b = [300] ->235500
  history = model.fit(x_train, y_train_scaled, epochs = 10, 
           validation_data = (x_valid_scaled, y_valid))
  
  # 查看训练参数变化
  type(history)
  history.history
  
  # 图像展示参数变化
  def plot_learning_curves(history):
      pd.DataFrame(history.history).plot(figsize = (8,5))
      plt.grid(True)
      plt.gca().set_ylim(0, 1)
      plt.show()
  plot_learning_curves(history)
  ```

- 数据归一化处理后，准确率上升。

### 2.3_实战-回调函数

- callbacks
  - tensorboard
  - EarlyStopping
  - ModelCheckpoint

## 3 神经网络讲解

- 神经网络
- 深度神经网络
  - 层次非常深的层次网络
- 激活函数
- 归一化与批归一化
  - 归一化：Min-Max归一化/ Z-score归一化
  - 批归一化：在每一层的输入上都做归一化处理
  - 为什么有效：
- Dropout：随机的丢弃随机单元
  - 作用：防止过拟合
    - 训练集上很好，测试集上不好
    - 参数太多，记住样本，不能泛化