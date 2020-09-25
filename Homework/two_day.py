import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #评估函数

# 导入鸢尾花数据集
irs_data = datasets.load_iris()
# Read Data and Xirs_target
Xirs_data = irs_data['data']
Xirs_target = irs_data['target']
# 对数据进行归一化处理
X_scaled = preprocessing.scale(Xirs_data)
# 拆分 训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Xirs_target ,test_size=0.2, random_state=5)
# 训练前的训练集和测试集
print('训练集:',x_train.max())
print('测试集',y_train.max())
#使用Svm 进行数据训练
clf = svm.SVC(C=1, kernel='linear')
clf.fit(x_train,y_train)
print('训练集:',x_train.min())
print('测试集',y_train.min())
# 统计预测正确的数量。
cs = clf.predict(x_test)
print('预测正确数量:',sum(cs == y_test))
#2.	统计预测错误的数量。
print("预测错误数量:", sum(cs != y_test))
#预测测试值
print('预测测试值:',clf.predict(x_test)[:10])
#准确率
print('训练集准确率:', accuracy_score(y_train, clf.predict(x_train)))
print('测试集准确率:', accuracy_score(y_test, clf.predict(x_test)))