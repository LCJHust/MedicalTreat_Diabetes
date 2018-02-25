# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import time

from feature_extract import make_data_feat

data_path = '../data/'

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='gb2312')

def ceate_feature_map(features):
    outfile = open('../cache/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def xgboost_make(train_feat, test_feat):  # xgboost训练及线下测评

    predictors = [f for f in test_feat.columns if f not in ['id', 'Blood_Sugar']]

    params = {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 2,
             'min_child_weight': 4, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'reg:linear'}

    num_round = 500
    params['nthread'] = 8
    params['metric'] = "rmse"

    num = 6
    train_preds = np.zeros(train_feat.shape[0])
    test_preds = np.zeros((test_feat.shape[0], num))
    kf = KFold(len(train_feat), n_folds=num, shuffle=True, random_state=3)
    test_feat = xgb.DMatrix(test_feat[predictors])
    for i, (train_index, test_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_feat1 = train_feat.iloc[train_index]
        train_feat2 = train_feat.iloc[test_index]
        xgb_train1 = xgb.DMatrix(train_feat1[predictors], train_feat1['Blood_Sugar'])
        xgb_train2 = xgb.DMatrix(train_feat2[predictors], train_feat2['Blood_Sugar'])
        evallist = [(xgb_train2, 'eval'), (xgb_train1, 'train')]
        bst = xgb.train(params, xgb_train1, num_round, evallist, early_stopping_rounds=100)

        features = [x for x in predictors]
        ceate_feature_map(features)
        importance = bst.get_fscore(fmap='../cache/xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        df = df.sort_values('fscore', ascending=False)
        #xgb特征重要性fscore
        train_preds[test_index] += bst.predict(xgb_train2)

        test_preds[:, i] = bst.predict(test_feat)

    print('xgb线下得分：    {}'.format(mean_squared_error(train_feat['Blood_Sugar'], train_preds / 1) * 0.5))

    test_preds = test_preds.mean(axis=1)
    submission = pd.DataFrame({'pred': test_preds})
    submission.to_csv('../sub/sub_xgb0.csv', index=False)

    return train_feat['Blood_Sugar'], train_preds/1






if __name__ == '__main__':

    t1 = time.time()

    train_feat, test_feat = make_data_feat()
    xgboost_make(train_feat, test_feat)

    t2 = time.time()

    print('用时{}秒'.format(t2 - t1))