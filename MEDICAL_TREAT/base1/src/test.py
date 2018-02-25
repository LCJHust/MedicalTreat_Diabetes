import pandas as pd
import numpy as np
import time
import datetime
from dateutil.parser import parse

#t0 = time.time()

data_path = '../data/'

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')
test1 = pd.read_csv('d://test.csv')

train_id = train.id.values.copy()
test_id = test.id.values.copy()
data = pd.concat([train, test])  # 训练集合测试集合并进行下一步处理



data['性别'] = data['性别'].map({'男': 1, '女': 0})

data1 = data['血红蛋白'] / data['红细胞计数']  # = 红细胞平均血红蛋白量
data1.columns = ['红细胞平均血红蛋白量']
data = pd.merge([data, data1, data1])


#data = pd.concat([data, data1], axis=1)
print(data.columns)


#data = pd.concat([data, data['*丙氨酸氨基转换酶']**2, data['*天门冬氨酸氨基转换酶']**2,
                     # data['肌酐']**2, data['尿素']**2, data['尿酸']**2], axis=1)


data = data.fillna(data.median(axis=0))  # 缺失值用中值填充
#data.to_csv('d://out.csv')



'''


train_feat = data[data.id.isin(train_id)]
test_feat = data[data.id.isin(test_id)]  # 做完预处理后，训练集和测试集分开

del train_feat['id']
#test_feat.to_csv('../data/test_feat1.csv')
train_preds = np.zeros(train_feat.shape[0])
test_preds0 = np.zeros((test_feat.shape[0], 5))
test_preds1 = np.zeros((test_feat.shape[1], 5))
print(test_preds1)
print(test_preds1.shape)
#print(test_preds1)





#train_preds.to_csv('../data/train_preds1.csv')

t1 = time.time()

print(t1-t0)
'''
