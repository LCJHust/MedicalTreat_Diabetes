# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time

train_data_file = '../data/d_train_20180102.csv'
test_data_file = '../data/d_test_A_20180102.csv'

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                  'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']


def mean_value():
    train_data = pd.read_csv(train_data_file, encoding="gb2312")
    train_data.columns = columns_rename
    train_data = train_data.fillna(0)
    aa = train_data[['TG', 'TC', 'Blood_Sugar']]
    value = np.mean(train_data['Blood_Sugar'])

    final = sum((train_data['Blood_Sugar'] - train_data['TC']) * (train_data['Blood_Sugar'] - train_data['TC'])) / (2 * len(train_data))
    print(final)

    # test_data = pd.read_csv(test_data_file, encoding="gb2312")
    # columns_rename.pop()
    # test_data.columns = columns_rename
    # test_data = test_data[['id']]
    # test_data['pred'] = value
    # test_data = test_data[['pred']]
    # test_data.to_csv('../sub/sub_mean.csv', index=False, header=False)


def data_corr():
    train_data = pd.read_csv(train_data_file, encoding="gb2312")
    train_data.columns = columns_rename
    del train_data['id']
    del train_data['Age']
    del train_data['Sex']
    del train_data['Date']
    corr = train_data.corr()
    print (0)




if __name__ == '__main__':

    t1 = time.time()

    mean_value()
    # data_corr()

    t2 = time.time()

    print ('time:', t2 - t1)
