import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score # 评估函数

abalone = pd.read_csv(r'Homework\data\abalone.csv')
# 查看前五行数据
print(abalone.head(5))
# 对当前数据进行探查
abalone.info()
# 获取前8列作为实际数据
x_data = abalone.iloc[ :, :-1]  # 实际数据
x_target = abalone.iloc[ :, -1]  # 数据标签--年龄
print(x_data.head(5))
print(x_target.head(5))

# 处理哑变量
Dv_data = abalone.iloc[ :, 0:1]
Dv = pd.get_dummies(Dv_data)
# 查看哑变量前五
print(Dv.head(5))
new_data = pd.concat([x_data, Dv], axis=1)
new_data.drop(columns=['Sex'], inplace=True) # 删除Sex列
# 划分训练集和测试集
x_data, x_test, y_data, y_test = train_test_split(new_data, x_target, test_size=0.2)

#  对数据进行归一化处理
scaler = StandardScaler()
x_train_data = scaler.fit_transform(x_data)
x_test_data = scaler.transform(x_test)

# 创建向量机模型
clf = svm.SVC()
clf.fit(x_train_data, y_data)
reslut = clf.predict(x_test_data)
# 查看预测测试集的结果
print(reslut[:10])

# Svm准确率分析
print('训练集准确率: {}'.format(accuracy_score(y_data, clf.predict(x_train_data))))
print('测试集准确率: {}'.format(accuracy_score(y_test, clf.predict(x_test_data))))