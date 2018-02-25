# -*- coding : utf-8 -*-
import numpy as np
import pandas as pd
from dateutil.parser import parse


def feature_extract():
    train_file = '../data20180128/d_train_20180102.csv'
    test_file = '../data20180128/d_test_A_20180102.csv'
    # test_file = '../data20180128/d_test_B_20180128.csv'


    train_data = pd.read_csv(train_file, encoding='gb2312')
    test_data = pd.read_csv(test_file, encoding='gb2312')

    """
    ['id', '性别', '年龄', '体检日期', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', 
    '甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', 
    '乙肝核心抗体', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', 
    '血小板平均体积', '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%', '血糖']
    """

    columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                      'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb',
                      'HBcAb', 'WBC', 'RBC', 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                      'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic', 'Basophil', 'Blood_Sugar']

    train_data.columns = columns_rename
    columns_rename.pop()
    test_data.columns = columns_rename

    train_id = train_data['id'].copy()
    test_id = test_data['id'].copy()

    data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    data['Date'] = (pd.to_datetime(data['Date'], format='%d/%m/%Y') - parse('2017-09-15')).dt.days
    data['Sex'] = data['Sex'].map({'男': 1, '女': 0})

    data = data.fillna(data.mean(axis=0))
    # data = (data-data.min()) / (data.max()-data.min())    #归一化

    data['TG/UA'] = data['TG'] / data['UA']
    data['ALT/AST'] = data['ALT'] / data['AST']
    data['MCV/MCHC'] = data['MCV'] / data['MCHC']
    data['Urea/UA'] = data['Urea'] / data['UA']
    data['Age*TG/UA'] = data['Age'] * data['TG/UA']
    data['Urea*TG/UA'] = data['Urea'] * data['TG/UA']

    data['RBC*MCHC'] = data['RBC'] * data['MCHC']
    data['RBC*MCV'] = data['RBC'] * data['MCV']

    del_col = ['Sex', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic', 'Basophil']
    columns = [x for x in data.columns if x not in del_col]
    data = data[columns]



    ##############加特征####################

    train_feat = data[data['id'].isin(train_id)]
    test_feat = data[data['id'].isin(test_id)]

    return train_feat, test_feat






if __name__ == '__main__':
    print(0)












