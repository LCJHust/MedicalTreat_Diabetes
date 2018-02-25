# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# import cPickle as pickle
import xgboost as xgb
import time
import matplotlib.pyplot as plt

from feature_extract import make_data_feat

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