# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from dateutil.parser import parse

train_data_file = '../data/d_train_20180102.csv'
test_data_file = '../data/d_test_A_20180102.csv'

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
    test_data = pd.read_csv(test_data_file, encoding='gb2312')
    test_data.columns = columns_rename

    train_id = train_data['id'].values.copy()
    test_id = test_data['id'].values.copy()
    data = pd.concat([train_data, test_data])

    data['Sex'] = data['Sex'].map({'男': 1, '女': 0})
    data['Date'] = (pd.to_datetime(data['Date'], format='%d/%m/%Y') - parse('2017-09-15')).dt.days

    data = data.fillna(data.mean(axis=0))
    # data['TG_UA'] = np.log1p(data['TG']) -  np.log1p(data['UA'])
    # data['ALT_AST'] = np.log1p(data['ALT']) - np.log1p(data['AST'])
    # data['MCV_MCHC'] = np.log1p(data['MCV']) - np.log1p(data['MCHC'])
    # data['Urea_UA'] = np.log1p(data['Urea']) - np.log1p(data['UA'])

    data['TG_UA'] = data['TG'] / data['UA']
    data['ALT_AST'] = data['ALT'] / data['AST']
    data['MCV_MCHC'] = data['MCV'] / data['MCHC']
    data['Urea_UA'] = data['Urea'] / data['UA']

    data['Age*TG/UA'] = data['Age'] * data['TG_UA']
    data['Urea*TG/UA'] = data['Urea'] * data['TG_UA']

    data['lg_ALT'] = np.log1p(data['ALT'])
    data['lg_Urea'] = np.log1p(data['Urea'])
    #
    # data['Age2'] = data['Age']**2
    # data['MCV_MCHC2'] = data['MCV_MCHC'] ** 2
    # data['TG_UA2'] = data['TG_UA']**2
    # data['Urea_UA2'] = data['Urea_UA']**2




 #   data['ALT'] = np.log1p(data['ALT'])
 #  data['lg_GGT'] = np.log1p(data['GGT'])
 #   data['lg_AST'] = np.log1p(data['AST'])
 #   data['lg_TC'] = np.log1p(data['TC'])
 #   data['lg_LDLC'] = np.log1p(data['LDL_C'])
 #   data['lg_HDLC'] = np.log1p(data['HDL_C'])
 #   data['lg_Cre'] = np.log1p(data['Cre'])
 #   data['lg_TG'] = np.log1p(data['TG'])

    # data['lg_Urea'] = np.log1p(data['Urea'])
    # del data['HBcAb']
    # del data['HBsAb']
    # del data['HBsAg']
    # del data['Acidophilic']
    # del data['Basophil']

#根据线性回归模型权重删除特征——效果变差，卒

#根据线性回归模型权重增加特征
    # data['TP_ALB'] = data['TP']/data['ALB']
    # data['TP_GLB'] = data['TP']/data['GLB']





    data_temp = (data - data.min()) / (data.max() - data.min())
    # data['temp'] = data_temp['MCV'] / data_temp['MCHC']
    # del data['TG_UA']
    # del data['Basophil']

    train_feat = data[data.id.isin(train_id)]
    # train_feat = train_feat[['temp', 'Age', 'Urea', 'TG_UA', 'Blood_Sugar']]
    # train_feat = train_feat.sort_values('Blood_Sugar', ascending=False)
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

if __name__ == '__main__':

    print (0)