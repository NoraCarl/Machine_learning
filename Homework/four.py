import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# 统计特征目标值的数量
print(df_pd_target.count())
Rlp_date = df_pd_data[['RM','LSTAT','PTRATIO']]
print(Rlp_date.shape)
# 绘制图像--绘制画布
plt.figure(figsize=(16,18))
for i in range(Rlp_date.columns.size):
    plt.subplot(2,2,i+1)
    plt.scatter(Rlp_date.iloc[:,i],df_pd_target)
    plt.title(Rlp_date.columns[i])

# 规划训练集和测试集 
x_data, x_test, y_data, y_test = train_test_split(df_pd_data,df_pd_target,test_size=0.2,random_state=5)
print(x_data.shape,y_data.shape,x_test.shape,y_test.shape)
# 训练模型 线性回归模型
Lr = LinearRegression(n_jobs=-1, normalize=True)
Lr.fit(x_data,y_data)
Prd = Lr.predict(x_test)
print('准确率:{}'.format(Lr.score(x_test,y_test)))

# 绘制图像 描述真实值和预测值
plt.figure(figsize=(14,8))
plt.plot(range(y_test.shape[0]),y_test,'-')
plt.plot(range(y_test.shape[0]),Prd,'--')
plt.legend(('真实值','预测值'),loc='upper right',fontsize=14)

plt.show()
