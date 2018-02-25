import pandas as pd
import numpy as np
import _pickle as pickle
import xgboost as xgb
import time
import matplotlib.pyplot as plt

from feature import feature_extract

# train_data_file = '../data20180128/d_train_20180102.csv'
# test_data_file = '../data20180128/d_test_A_20180102.csv'
#
# columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
#                   'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
#                   'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
#                   'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']

def compare():
    sub0 = pd.read_csv('../data20180128/d_answer_a_20180128.csv')
    sub0.columns = ['pred0']
    sub1 = pd.read_csv('../sub/sub_final0.csv')
    sub1.columns = ['pred1']
    result = pd.concat([sub0, sub1], axis=1)
    result = result.sort_values('pred0', ascending=False)
    result['diff'] = result['pred1'] - result['pred0']
    print (result['pred0'].mean(), result['pred0'].std())
    print (result['pred1'].mean(), result['pred1'].std())
    final = sum((result['pred0'] - result['pred1']) * (result['pred0'] - result['pred1'])) / (2 * len(result))
    print(final)

if __name__ == '__main__':

    t1 = time.time()

    # mean_value()
    # data_corr()
    compare()
    # train_days_count()
    # test_days_count()
    # off_line()
    # result_add()
    # plot_features()

    t2 = time.time()

    print ('time:', t2 - t1)