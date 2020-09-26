import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib.numpy_pickle import dump
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

# 防中文乱码
plt.rcParams['font-sans-serif']=[u'SimHei']
plt.rcParams['axes.unicode_minus']=False

digits = datasets.load_digits()
dis_data = digits['data']
dis_target = digits['target']
# 获取当前数据的所有键值
print(digits.keys())
# 查看图片数据维度
print(digits['images'].shape)
# 对图片数据进行展开 使用reshape对格式控制
print(dis_data[0].reshape(8,8))
# 查看第一张图的数据
print(digits['images'][0])

# 绘制第一张图 图像
plt.imshow(digits['images'][0])

# 绘制1-9的图形
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    ax[i].imshow(dis_data[i].reshape((8,8)), cmap='Greys', interpolation='nearest')

# 对训练集和测试集进行划分 使用random_state进行调试参数
x_data, x_test, y_data, y_test = train_test_split(digits['data'], digits['target'], test_size=0.25, random_state=33)
# 分别查看训练集和测试集的形状大小
print(x_data.shape,x_test.shape)

# 开始建立向量机模型
clf = svm.SVC(gamma='scale')
clf.fit(x_data,y_data)
reslut = clf.predict(x_test)
# 生成新列表  使用zip将numpy.ndarray进行重新组合生成新的元组  
images_array = list(zip(x_test.reshape((450,8,8)),reslut))

#开始绘制图像 建立画布
plt.figure(figsize=(16,9))
for x,y in enumerate(images_array[:20]):
    # 这里需要对多张图片进行绘制--子图
    plt.subplot(4,5,x+1)
    plt.imshow(y[0], cmap='Greys', interpolation='nearest')
    plt.title("value:{}".format(y[1]))
plt.show()

# 模型保存
dump(clf,'Model.mx')