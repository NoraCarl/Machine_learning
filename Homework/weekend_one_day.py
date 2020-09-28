import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA  # 导入将维模块
from sklearn import preprocessing # 导入归一化模块
from sklearn import svm  # 导入向量机
from sklearn.metrics import recall_score,accuracy_score # 导入评估模块

irs = datasets.load_iris()
irs_data = irs['data']
irs_target = irs['target']
irs_target_names = irs['target_names']

# 归一化处理
x_scale = preprocessing.scale(irs_data)
# 划分训练集和测试集
x_data, x_test, y_data, y_test = train_test_split(x_scale, irs_target,test_size=0.2,random_state=10)
# 将训练集从4维降到3维
pca = PCA(n_components=3)
x_data = pca.fit_transform(x_data)
x_test = pca.transform(x_test)
# 训练数据
clf = svm.SVC(C=10, gamma='scale', decision_function_shape='ovr')
clf.fit(x_data, y_data)
reslut = clf.predict(x_test)
print("预测值:{}".format(reslut))
clf.score(x_data,y_data)
print('{}'.format(accuracy_score(y_data, clf.predict(x_data))))
print('{}'.format(accuracy_score(y_test, clf.predict(x_test))))
print('{}'.format(clf.decision_function(x_data)))