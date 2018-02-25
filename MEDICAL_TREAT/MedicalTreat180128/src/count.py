# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# import cPickle as pickle
import xgboost as xgb
import time
import matplotlib.pyplot as plt

from feature_extract import make_data_feat


train_data_file = '../data/d_train_20180102.csv'
test_data_file = '../data/d_test_A_20180102.csv'

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                  'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']


def mean_value():
    train_feat, test_feat = make_data_feat()

    train_feat = train_feat.sort_values('Blood_Sugar', ascending=False)

    train1 = train_feat[0:1410]
    train2 = train_feat[1410:2820]
    train3 = train_feat[2820:4230]
    train4 = train_feat[4230:5640]

    mean1 = train1.mean().to_frame()
    mean2 = train2.mean().to_frame()
    mean3 = train3.mean().to_frame()
    mean4 = train4.mean().to_frame()

    result = pd.DataFrame({'feat': mean1.index})
    result['1'] = mean1.values
    result['2'] = mean2.values
    result['3'] = mean3.values
    result['4'] = mean4.values

    print (0)

def data_corr():
    train_feat, test_feat = make_data_feat()
    del train_feat['id']
    corr = train_feat.corr()

    train_feat_temp = (train_feat - train_feat.min()) / (train_feat.max() - train_feat.min())
    corr_temp = train_feat_temp.corr()

    print (0)


def compare():
    sub0 = pd.read_csv('../sub/sub7802.csv')
    sub0.columns = ['pred0']
    sub1 = pd.read_csv('../sub/sub_final22.csv')
    sub1.columns = ['pred1']
    result = pd.concat([sub0, sub1], axis=1)
    result = result.sort_values('pred0', ascending=False)
    result['diff'] = result['pred1'] - result['pred0']
    print (result['pred0'].mean(), result['pred0'].std())
    print (result['pred1'].mean(), result['pred1'].std())
    final = sum((result['pred0'] - result['pred1']) * (result['pred0'] - result['pred1'])) / (2 * len(result))
    print(final)


def train_days_count():
    train_data = pd.read_csv(train_data_file, encoding="gb2312")
    train_data.columns = columns_rename
    train_data = train_data[['Date', 'Blood_Sugar']]
    # time0 = ['01/09/2017'] * len(train_data)
    # time0 = pd.to_datetime(pd.Series(time0), format='%d/%m/%Y')
    train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%Y')
    # train_data['Date'] = train_data['Date'].values - time0.values
    # train_data['Date'] = np.array(train_data['Date']) / np.timedelta64(24, 'h')
    # train_data['Date'] = np.int32(train_data['Date'])
    num = train_data['Date'].value_counts()
    num = num.to_frame()
    num['Date_real'] = num.index

    mean = train_data.groupby('Date', as_index=False).mean()
    mean = mean.sort_values('Date')
    print (0)


def test_days_count():
    test_data = pd.read_csv(test_data_file, encoding="gb2312")
    temp_rename = columns_rename[0:41]
    test_data.columns = temp_rename
    test_data = test_data[['Date']]
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d/%m/%Y')
    num = test_data['Date'].value_counts()
    num = num.to_frame()
    num['Date_real'] = num.index

    print(0)


def off_line():

    trainData = make_train_data()  # 提取特征
    label = trainData[['Blood_Sugar']]  # 获得标签

    del trainData['id']
    del trainData['Blood_Sugar']
    trainData = trainData.values
    label = label.values

    result = pd.DataFrame(label, columns=['label'])
    for i in range(0, 2):
        model = 'xgboost' + str(i)
        bst = pickle.load(open(model, 'rb'))  # 线下测评
        dtest = xgb.DMatrix(trainData)
        pred = bst.predict(dtest)
        predfeat = 'pred' + str(i)
        result[predfeat] = pred
    # result['pred'] = (result['pred0'] + result['pred1'] + result['pred2'] +
    #                   result['pred3'] + result['pred4'] + result['pred5'] + result['pred6']) / 7
    result['pred0'] = result['pred0'] * (1 + (result['pred0'] - result['pred0'].mean()) / 50)
    result['pred1'] = result['pred1'] * (1 + (result['pred1'] - result['pred1'].mean()) / 50)
    result['pred'] = (result['pred0'] + result['pred1']) / 2

    final = sum((result['pred0'] - result['label']) * (result['pred0'] - result['label'])) / (2 * len(result))
    print(final)
    final = sum((result['pred1'] - result['label']) * (result['pred1'] - result['label'])) / (2 * len(result))
    print(final)
    final = sum((result['pred'] - result['label']) * (result['pred'] - result['label'])) / (2 * len(result))
    print(final)


def result_deal():
    result = pd.read_csv('../sub/sub8707.csv')
    result['pred'] = result['pred'] * (1 + (result['pred'] - result['pred'].mean()) / 50)


def result_add():
    sub0 = pd.read_csv('../sub/sub_lgb0.csv')
    sub0.columns = ['pred0']
    sub1 = pd.read_csv('../sub/sub_xgb0.csv')
    sub1.columns = ['pred1']

    result = pd.concat([sub0, sub1], axis=1)

    result['pred'] = result['pred0']*0.8 + result['pred1']*0.2
    result = result[['pred']]
    result.to_csv('../sub/sub_final0.csv', index=False)


def plot_features():
    train_feat, test_feat = make_data_feat()
    aa = train_feat['Urea'].value_counts()
    aa = aa.to_frame()
    aa['num'] = aa.index
    # plt.hist(train_feat['Age'], 100)
    import math
    train_feat['TG/UA'] = np.exp(train_feat['TG/UA'])

    plt.scatter(train_feat['TG/UA'], train_feat['Blood_Sugar'])
    plt.show()






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
