# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import time

from feature_extract import make_data_feat

train_data_file = '../data/d_train_20180102.csv'
test_data_file = '../data/d_test_A_20180102.csv'

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC', 'RBC', 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                  'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']

train_data = pd.read_csv(train_data_file, encoding='gb2312')
train_data.columns = columns_rename
columns_rename.pop()    #血糖值出栈
test_data = pd.read_csv(test_data_file, encoding='gb2312')
test_data.columns = columns_rename

train_id = train_data['id'].values.copy()
test_id = test_data['id'].values.copy()


data = pd.concat([train_data, test_data])

train_feat = data[data.id.isin(train_id)]

test_feat = data[data.id.isin(test_id)]
print(test_feat.columns)

predictors = [f for f in test_feat.columns if f not in ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' , 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC', 'RBC', 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                  'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']]
print(predictors)

