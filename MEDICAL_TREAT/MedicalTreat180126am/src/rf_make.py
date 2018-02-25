# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cross_validation import KFold
# from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import time

from feature_extract import make_data_feat

def randomforest_make(train_feat, test_feat):  # randomforest训练

    predictors = [f for f in test_feat.columns if f not in ['id', 'Blood_Sugar']]
    # train_feat[predictors] = (train_feat[predictors] - train_feat[predictors].min()) / (
    # train_feat[predictors].max() - train_feat[predictors].min())
    # train_feat['TG'] = train_feat['TG'] * 10

    # rf = RandomForestRegressor(max_depth=5, n_estimators=300, random_state=2018)
    from sklearn import linear_model
    # rf = linear_model.LinearRegression()
    rf = linear_model.Ridge(alpha=0.05, normalize=True)

    num = 6
    train_preds = np.zeros(train_feat.shape[0])
    test_preds = np.zeros((test_feat.shape[0], num))
    kf = KFold(len(train_feat), n_folds=num, shuffle=True, random_state=0)
    #kf = KFold(n_splits=len(train_feat), n_folds=num, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_feat1 = train_feat.iloc[train_index]
        train_feat2 = train_feat.iloc[test_index]
        rf.fit(train_feat1[predictors], train_feat1['Blood_Sugar'])
        train_preds[test_index] += rf.predict(train_feat2[predictors])
        # train_preds += rf.predict(train_feat[predictors])
        test_preds[:, i] = rf.predict(test_feat[predictors])

    print('rf线下得分：    {}'.format(mean_squared_error(train_feat['Blood_Sugar'], train_preds/1) * 0.5))

    # train_feat['pred'] = train_preds
    # train_feat = train_feat[['Blood_Sugar', 'pred']].sort_values('Blood_Sugar', ascending=False)
    # train_feat = train_feat[0:100]
    # print('线下得分：    {}'.format(mean_squared_error(train_feat['Blood_Sugar'], train_feat['pred']) * 0.5))

    submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
    submission.to_csv('../sub/sub_rf0.csv', index=False)

    return train_feat['Blood_Sugar'], train_preds / 1


if __name__ == '__main__':

    t1 = time.time()

    train_feat, test_feat = make_data_feat()
    randomforest_make(train_feat, test_feat)

    t2 = time.time()

    print('用时{}秒'.format(t2 - t1))

