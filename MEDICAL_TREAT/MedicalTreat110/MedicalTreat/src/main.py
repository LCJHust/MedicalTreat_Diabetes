# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import time

from xgb_make import xgboost_make
from lgb_make import lightgbm_make
from feature_extract import make_data_feat
import datetime

if __name__ == '__main__':

    t1 = time.time()

    train_feat, test_feat = make_data_feat()

    lightgbm_make(train_feat, test_feat)
    xgboost_make(train_feat, test_feat)

    sub0 = pd.read_csv('../sub/sub_lgb0.csv')
    sub0.columns = ['pred0']
    sub1 = pd.read_csv('../sub/sub_xgb0.csv')
    sub1.columns = ['pred1']

    result = pd.concat([sub0, sub1], axis=1)

    result['pred'] = result['pred0'] * 0.8 + result['pred1'] * 0.2
    result = result[['pred']]
#    result.to_csv('../sub/sub_final0.csv', index=False)
    result.to_csv(r'../sub/sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                      index=False, float_format='%.4f')

    t2 = time.time()

    print('time:', t2 - t1)




