# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import time

from xgb_make import xgboost_make
from lgb_make import lightgbm_make
from rf_make import randomforest_make
from feature_extract import make_data_feat

if __name__ == '__main__':

    t1 = time.time()

    train_feat, test_feat = make_data_feat()

    lgb_label, lgb_pred = lightgbm_make(train_feat, test_feat)
    xgb_label, xgb_pred =  xgboost_make(train_feat, test_feat)
    rf_label, rf_pred = randomforest_make(train_feat, test_feat)

    print('线下得分：    {}'.format(mean_squared_error(lgb_label, lgb_pred) * 0.5))
    print('线下得分：    {}'.format(mean_squared_error(xgb_label, xgb_pred) * 0.5))
    print('线下得分：    {}'.format(mean_squared_error(xgb_label, ((lgb_pred*0.8 + xgb_pred*0.2)*1 + rf_pred*0)) * 0.5))
    #
    # df = lgb_label.to_frame()
    # df['pred'] = (lgb_pred * 0.8 + xgb_pred * 0.2)*0.9 + rf_pred*0.1
    # df = df.sort_values('pred', ascending=True)


    sub0 = pd.read_csv('../sub/sub_lgb0.csv')
    sub0.columns = ['pred0']
    sub1 = pd.read_csv('../sub/sub_xgb0.csv')
    sub1.columns = ['pred1']
    sub2 = pd.read_csv('../sub/sub_rf0.csv')
    sub2.columns = ['pred2']

    result = pd.concat([sub0, sub1, sub2], axis=1)

    result['pred'] = (result['pred0'] * 0.8 + result['pred1'] * 0.2)* 1 + result['pred2']* 0
    result[result >= 7.8] = result[result >= 7.8] + (result[result >= 7.8] - 7.5) * 4
    result = result[['pred']]
    result.to_csv('../sub/sub_final2.csv', index=False)

    t2 = time.time()

    print ('time:', t2 - t1)
#



