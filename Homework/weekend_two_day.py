import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import metrics #评估
from sklearn.cluster import KMeans  # 导入K-means
from sklearn import preprocessing # 导入归一化
from sklearn.metrics import fowlkes_mallows_score # 聚类模型评估


irs = datasets.load_iris()
irs_data = irs['data']
irs_target = irs['target']
irs_feature_names = irs['feature_names']

# 迭代鸢尾花2-6类
for i in range(2, 7):
    # 使用K-means 训练模型
    Km = KMeans(n_clusters=i, init='k-means++', random_state=5, algorithm='auto')
    Km.fit(irs_data)
    Fms = fowlkes_mallows_score(irs_target, Km.labels_)
    print('FMI{}:{}'.format(i,Fms))

# 使用聚类训练算法 预测
Kms = KMeans(n_clusters=5, init='k-means++', random_state=10)
Kms.fit(irs_data)

### 效果评估
score_funcs = [
    metrics.adjusted_rand_score,#ARI
    metrics.v_measure_score,#均一性和完整性的加权平均
    metrics.adjusted_mutual_info_score,#AMI
    metrics.mutual_info_score,#互信息
]
# Ari
print(metrics.adjusted_rand_score(irs_target, Kms.predict(irs_data)))
# Ami
print(metrics.adjusted_mutual_info_score(irs_target, Kms.predict(irs_data)))
# 均一性和完整性
print(metrics.v_measure_score(irs_target, Kms.predict(irs_data)))
# 互相信
print(metrics.mutual_info_score(irs_target, Kms.predict(irs_data)))