import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
dis_data = digits['data']
dis_target = digits['target']

df_dis = pd.DataFrame(data=dis_data, columns=digits['feature_names'])
print(df_dis.info)