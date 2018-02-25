# -*- coding : utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame
from dateutil.parser import parse


train_file = '../data20180128/d_train_20180102.csv'
testA_file = '../data20180128/d_test_A_20180102.csv'
testA_answer_file = '../data20180128/d_answer_a_20180128.csv'
testB_file = '../data20180128/d_test_B_20180128.csv'


train_data = pd.read_csv(train_file, encoding='gb2312')
testA_data = pd.read_csv(testA_file, encoding='gb2312')
testB_data = pd.read_csv(testB_file, encoding='gb2312')
# testA_answer = pd.read_csv(testA_answer_file, encoding='gb2312')
# trainA_data = pd.concat([testA_data, testA_answer], axis=1, ignore_index=True)
# train_all = pd.concat([train_data, trainA_data], axis=0, ignore_index=True)
# print(train_data.shape)
# print(trainA_data.shape)
#
# print(train_all.head(5))

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                      'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb',
                      'HBcAb', 'WBC', 'RBC', 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                      'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic', 'Basophil', 'Blood_Sugar']

train_data.columns = columns_rename
columns_rename.pop()
testA_data.columns = columns_rename
testB_data.columns = columns_rename

train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%Y')
testA_data['Date'] = pd.to_datetime(testA_data['Date'], format='%d/%m/%Y')
testB_data['Date'] = pd.to_datetime(testB_data['Date'], format='%d/%m/%Y')

print(train_data['Date'].min(), train_data['Date'].max())
print(testA_data['Date'].min(), testA_data['Date'].max())
print(testB_data['Date'].min(), testB_data['Date'].max())

train_data['Date'] = (pd.to_datetime(train_data['Date'], format='%d/%m/%Y') - parse('2017-09-15')).dt.days
train_data['Sex'] = train_data['Sex'].map({'男': 1, '女': 0})
testA_data['Date'] = (pd.to_datetime(testA_data['Date'], format='%d/%m/%Y') - parse('2017-09-15')).dt.days
testA_data['Sex'] = testA_data['Sex'].map({'男': 1, '女': 0})
testB_data['Date'] = (pd.to_datetime(testB_data['Date'], format='%d/%m/%Y') - parse('2017-09-15')).dt.days
testB_data['Sex'] = testB_data['Sex'].map({'男': 1, '女': 0})

print(train_data['Date'].min(), train_data['Date'].max())
print(testA_data['Date'].min(), testA_data['Date'].max())
print(testB_data['Date'].min(), testB_data['Date'].max())

# test_date = (pd.to_datetime(train_data.loc[Q], format='%d/%m/%Y') - parse('2017-09-15')).dt.days
# print(test_date)
#
#
# # mean = (testA_data.mean(axis=0), testB_data.mean(axis=0))
# #print(testA_data.mean(axis=0).shape)
#
# mean = pd.concat([testA_data.mean(axis=0), testB_data.mean(axis=0)], axis=1, ignore_index=True)
# median = pd.concat([testA_data.median(axis=0), testB_data.median(axis=0)], axis=1, ignore_index=True)
# max = pd.concat([testA_data.max(axis=0), testB_data.max(axis=0)], axis=1, ignore_index=True)
# min = pd.concat([testA_data.min(axis=0), testB_data.min(axis=0)], axis=1, ignore_index=True)
# var = pd.concat([testA_data.var(axis=0), testB_data.var(axis=0)], axis=1, ignore_index=True)
# print(mean,var)