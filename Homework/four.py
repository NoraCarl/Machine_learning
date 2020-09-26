import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split


# 解决中文乱码
plt.rcParams['font.sans-serif']=[u'SimHei']
plt.rcParams['axes.unicode_minus']=False

# 导入波士顿房价数据集
Boston = datasets.load_boston()
# 获取当前数据和目标
bt_data = Boston['data']
bt_target = Boston['target']
bt_feature_names = Boston['feature_names']
# 获取所有Keys
print('获取Kyes:{}'.format(Boston.keys()))
# 查看数据的形状
print('查看数据的形状:{}{}{}'.format(bt_data.shape, bt_target.shape, bt_feature_names.shape))
# 转换DataFrame格式
df_pd_data = pd.DataFrame(data=bt_data, columns=bt_feature_names)
df_pd_target = pd.DataFrame(data=bt_target, columns=['target'])

# 开始绘制图像--画布
plt.figure(figsize=(18,24))
for i in range(bt_feature_names.size):
    plt.subplot(5,3,i+1)
    plt.scatter(df_pd_data.iloc[:,i],df_pd_target)
    plt.title(bt_feature_names[i],color='red')

plt.show()