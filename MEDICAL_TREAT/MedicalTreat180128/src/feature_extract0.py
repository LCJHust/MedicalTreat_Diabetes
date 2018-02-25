# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from dateutil.parser import parse
from scipy.stats import mode

train_data_file = '../data/d_train_20180102.csv'
test_data_file = '../data/d_test_B_20180128.csv'
# test_data_file = '../data/d_test_A_20180102.csv'

"""
['id', '性别', '年龄', '体检日期', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', 
'甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', 
'乙肝核心抗体', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', 
'血小板平均体积', '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%', '血糖']
"""

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT',
                  'MPV', 'PDW','PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']


def make_data_feat():
    train_data = pd.read_csv(train_data_file, encoding='gbk')
    train_data.columns = columns_rename
    columns_rename.pop()
    test_data = pd.read_csv(test_data_file, encoding='gbk')
    test_data.columns = columns_rename

    train_id = train_data['id'].values.copy()
    test_id = test_data['id'].values.copy()
    data = pd.concat([train_data, test_data])


    data['Sex'] = data['Sex'].map({'男':1, '女':0})         #???
    data['Date'] = (pd.to_datetime(data['Date'], format='%d/%m/%Y') - parse('2017-09-15')).dt.days

    ###########################echo修改####################################

    # data['HbeAg']=data['HbeAg'].fillna(mode(data['HbeAg']),inplace=False)#众数
    # data = data.fillna(mode(data)) #axis=0表示列，1表示行，median表示中位数
    data = data.fillna(data.median(axis=0))
    ###############################################################

    ###########################echo修改   0.7848  20180126pm   ####################################
    # data = data.fillna(0.5*(data.mean(axis=0))+0.5*(data.median(axis=0)),inplace=False) #axis=0表示列，1表示行，median表示中位数
    # # data = data.fillna(data.median(axis=0))
    ###############################################################

    data['TG/UA'] = data['TG'] / data['UA']
    data['ALT/AST'] = data['ALT'] / data['AST']
    data['MCV/MCHC'] = data['MCV'] / data['MCHC']
    data['Urea/UA'] = data['Urea'] / data['UA']
    data['Age*TG/UA'] = data['Age'] * data['TG/UA']
    data['Urea*TG/UA'] = data['Urea'] * data['TG/UA']
    # del data['Sex']

    ###########################echo修改####################################
    # # data['nPLT*MPV'] = np.log1p(data['PLT']*data['MPV'])
    # data['PLT*MPV'] = data['PLT']*data['MPV']
    # # data['nPLT*PDW'] = np.log1p(data['PLT']*data['PDW'])
    # data['PLT*PDW'] = data['PLT']*data['PDW']
    # # data['nPLT*PCT'] = np.log1p(data['PLT']*data['PCT'])
    # data['PLT*PCT'] = data['PLT']*data['PCT']
    # data['nTC'] = np.log1p(data['TC'])
    #
    # data['RBC*MCHC'] = data['RBC'] * data['MCHC']
    # data['RBC*MCV'] = data['RBC'] * data['MCV']
    # # data['nRBC*MCHC'] = np.log1p(data['RBC']*data['MCHC'])
    # # data['nRBC*MCV'] = np.log1p(data['RBC']*data['MCV'])
    # #'TP'
    # del_col = [ 'ALB', 'GLB', 'Sex', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb', 'Urea', 'Cre', 'UA','Monocytes']
    # # del_col = ['Sex', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb','Neutrophil', 'Lymph', ,]
    # columns = [x for x in data.columns if x not in del_col]
    # data = data[columns]
    ###############################################################
    ###########################echo修改   final14####################################
    # # data['nPLT*MPV'] = np.log1p(data['PLT']*data['MPV'])
    # data['PLT*MPV'] = data['PLT']*data['MPV']
    # # data['nPLT*PDW'] = np.log1p(data['PLT']*data['PDW'])
    # data['PLT*PDW'] = data['PLT']*data['PDW']
    # # data['nPLT*PCT'] = np.log1p(data['PLT']*data['PCT'])
    # data['PLT*PCT'] = data['PLT']*data['PCT']
    # data['nTC'] = np.log1p(data['TC'])
    #
    # data['RBC*MCHC'] = data['RBC'] * data['MCHC']
    # data['RBC*MCV'] = data['RBC'] * data['MCV']
    # # data['nRBC*MCHC'] = np.log1p(data['RBC']*data['MCHC'])
    # # data['nRBC*MCV'] = np.log1p(data['RBC']*data['MCV'])
    # #'TP'
    # del_col = [ 'ALB', 'GLB', 'Sex', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb', 'Urea', 'Cre', 'UA','Monocytes']
    # # del_col = ['Sex', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb','Neutrophil', 'Lymph', ,]
    # columns = [x for x in data.columns if x not in del_col]
    # data = data[columns]
    ###############################################################
    ###########################echo修改   final13####################################
    # data['RBC*MCHC'] = data['RBC'] * data['MCHC']
    # data['RBC*MCV'] = data['RBC'] * data['MCV']
    # # data['nRBC*MCHC'] = np.log1p(data['RBC']*data['MCHC'])
    # # data['nRBC*MCV'] = np.log1p(data['RBC']*data['MCV'])
    # # 'TP'
    # del_col = ['ALB', 'GLB', 'Sex', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb', 'Urea', 'Cre', 'UA', 'Monocytes']
    # # del_col = ['Sex', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb','Neutrophil', 'Lymph', ,]
    # columns = [x for x in data.columns if x not in del_col]
    # data = data[columns]
    ###############################################################

    #  ###########################echo修改   0.7848  20180126pm   ####################################
    # # data['RBC*MCH'] = data['RBC']*data['MCH']
    # data['RBC*MCHC'] = data['RBC']*data['MCHC']
    # data['RBC*MCV'] = data['RBC']*data['MCV']
    #
    # # 'Age', 'Date','Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil'
    # del_col = ['Sex','HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb','HBcAb']
    # columns = [x for x in data.columns if x not in del_col]
    # data = data[columns]
    ###############################################################


    ############################echo修改  0.7819	20180126am  ####################################
    # data['RBC*MCHC'] = data['RBC']*data['MCHC']
    # data['RBC*MCV'] = data['RBC']*data['MCV']
    #
    # # 'Age', 'Date','Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil'
    # del_col = ['TP', 'ALB', 'GLB','Sex','HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb','HBcAb','Urea', 'Cre']
    # columns = [x for x in data.columns if x not in del_col]
    # data = data[columns]

    ################################################################

    # ############################echo修改45 / 0.7802    20180125pm   ####################################
    # data['RBC*MCH'] = data['RBC']*data['MCH']
    data['RBC*MCHC'] = data['RBC'] * data['MCHC']
    data['RBC*MCV'] = data['RBC'] * data['MCV']

    # 'Age', 'Date','Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil'
    del_col = ['TP', 'ALB', 'GLB', 'Sex', 'HBsAg', 'HBsAb', 'HbeAg', 'HBeAb', 'HBcAb', 'Urea', 'Cre', 'UA']
    columns = [x for x in data.columns if x not in del_col]
    data = data[columns]
    # ################################################################


    # data['temp2'] = data['Age'] * data['MCV/MCHC']
    # data['MCV_MCHC2'] = data['MCV/MCHC'] ** 2
    # data['TG_UA2'] = data['TG/UA'] ** 2
    # data['Urea_UA2'] = data['Urea/UA'] ** 2
    # data['TG2'] = data['TG'] ** 2
    # data['Urea2'] = data['Urea'] ** 2
    # data['UA2'] = data['UA'] ** 2
    # data['AST2'] = data['AST'] ** 2
    # data['temp2'] = (data['LDL_C'] ** 3) / (data['HDL_C'] ** 2)

    data_temp = (data - data.min()) / (data.max() - data.min())     #归一化

    # data['temp'] = 2*data_temp['Age'] + 2*data_temp['ALP'] + data_temp['ALT'] + 2*data_temp['AST'] + data_temp['Basophil'] + \
    #                data_temp['Cre'] + 2*data_temp['GGT'] + data_temp['HGB'] + data_temp['MCH'] + data_temp['MCHC'] + \
    #                data_temp['MCV'] + data_temp['PCV'] + data_temp['RBC'] + data_temp['Sex'] + data_temp['TC'] + \
    #                2*data_temp['TG'] + data_temp['TP'] + 2*data_temp['UA'] + 2*data_temp['Urea']
    # data['temp2'] = 2*data_temp['HDL_C'] + data_temp['PCT'] + data_temp['PLT'] + data_temp['RDW']
    # data['temp3'] = data['temp'] / data['temp2']
    # data['temp2'] = data_temp['Age'] + data_temp['ALP'] + data_temp['ALT'] + data_temp['AST'] + data_temp['Basophil'] + \
    #                data_temp['Cre'] + data_temp['GGT'] + data_temp['HGB'] + data_temp['MCH'] + data_temp['MCHC'] + \
    #                data_temp['MCV'] + data_temp['PCV'] + data_temp['RBC'] + data_temp['TC'] + \
    #                data_temp['TG'] + data_temp['TP'] + data_temp['UA'] + data_temp['Urea']
    # data['lg_ALT'] = np.log1p(data['ALT'])
    # data['GGT'] = np.log1p(data['GGT'])
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

if __name__ == '__main__':

    print (0)