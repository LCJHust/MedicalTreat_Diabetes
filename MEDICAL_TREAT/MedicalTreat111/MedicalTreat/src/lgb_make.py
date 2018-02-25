# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import time

from feature_extract import make_data_feat


def lightgbm_make(train_feat, test_feat):

    predictors = [f for f in test_feat.columns if f not in ['id', 'Blood_Sugar']]

    params = {'learning_rate': 0.01, 'boosting_type': 'gbdt', 'objective': 'regression',
        'metric': 'mse', 'sub_feature': 0.7, 'num_leaves': 10, 'colsample_bytree': 0.7,
        'feature_fraction': 0.7, 'min_data': 100, 'min_hessian': 1, 'verbose': -1}

    num = 6
    train_preds = np.zeros(train_feat.shape[0])
    test_preds = np.zeros((test_feat.shape[0], num))
    kf = KFold(len(train_feat), n_folds=num, shuffle=True, random_state=3)
    for i, (train_index, test_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_feat1 = train_feat.iloc[train_index]
        train_feat2 = train_feat.iloc[test_index]
        lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['Blood_Sugar'])
        lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['Blood_Sugar'])
        gbm = lgb.train(params, lgb_train1, num_boost_round=3000, valid_sets=[lgb_train1, lgb_train2],
                        verbose_eval=100, early_stopping_rounds=100)
        feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
        # train_preds[test_index] += gbm.predict(train_feat2[predictors])
        train_preds += gbm.predict(train_feat[predictors])
        test_preds[:, i] = gbm.predict(test_feat[predictors])

    print('线下得分：    {}'.format(mean_squared_error(train_feat['Blood_Sugar'], train_preds/num) * 0.5))

    final_lgb = mean_squared_error(train_feat['Blood_Sugar'], train_preds/num) * 0.5

    submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
    submission.to_csv('../sub/sub_lgb0.csv', index=False)

    # return train_feat['Blood_Sugar'], train_preds
    return final_lgb


if __name__ == '__main__':

    t1 = time.time()

    train_feat, test_feat = make_data_feat()
    lightgbm_make(train_feat, test_feat)

    t2 = time.time()

    print('用时{}秒'.format(t2 - t1))
