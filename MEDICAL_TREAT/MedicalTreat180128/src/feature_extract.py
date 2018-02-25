# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from dateutil.parser import parse

train_data_file = '../data/d_train_20180102.csv'
test_data_file = '../data/d_test_A_20180102.csv'
test_data_fileA = '../data/d_test_A_20180102.csv'
real_Blood_SugarA = '../data/d_answer_a_20180128.csv'

"""
['id', '性别', '年龄', '体检日期', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', 
'甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', 
'乙肝核心抗体', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', 
'血小板平均体积', '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%', '血糖']
"""

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                  'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']


def make_data_feat():

    train_data = pd.read_csv(train_data_file, encoding='gb2312')
    train_data.columns = columns_rename
    columns_rename.pop()

    # extra_data = pd.read_csv(test_data_fileA, encoding='gb2312')
    # extra_data.columns = columns_rename
    # extra_Blood = pd.read_csv(real_Blood_SugarA, encoding='gb2312')
    # extra_data['Blood_Sugar'] = extra_Blood.values
    # train_data = train_data.append(extra_data)

    test_data = pd.read_csv(test_data_file, encoding='gb2312')
    test_data.columns = columns_rename

    train_id = train_data['id'].values.copy()
    test_id = test_data['id'].values.copy()
    data = pd.concat([train_data, test_data])

    data['Sex'] = data['Sex'].map({'男': 1, '女': 0})
    data['Date'] = (pd.to_datetime(data['Date'], format='%d/%m/%Y') - parse('2017-09-15')).dt.days

    data = data.fillna(data.median(axis=0))

    data['TG/UA'] = data['TG'] / data['UA']
    data['ALT/AST'] = data['ALT'] / data['AST']
    data['MCV/MCHC'] = data['MCV'] / data['MCHC']
    data['Urea/UA'] = data['Urea'] / data['UA']
    data['Age*TG/UA'] = data['Age'] * data['TG/UA']
    data['Urea*TG/UA'] = data['Urea'] * data['TG/UA']

    data['RBC*MCHC'] = data['RBC'] * data['MCHC']
    data['RBC*MCV'] = data['RBC'] * data['MCV']


    del_col = ['TP', 'ALB', 'GLB', 'Sex', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb', 'Urea', 'Cre', 'UA']
    columns = [x for x in data.columns if x not in del_col]
    data = data[columns]


    # del_col = ['Sex', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic', 'Basophil']
    # data = data.drop(columns=del_col)


    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

if __name__ == '__main__':

    print (0)


    train.head(10)
    train.shape()
    train.info()
    train.isnull().sum(axis=0)
    np.setdiff1d(train_features, test_features)
