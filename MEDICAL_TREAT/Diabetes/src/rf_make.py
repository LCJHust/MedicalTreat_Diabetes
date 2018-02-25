import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn import linear_model
import time

from feature import feature_extract

def randomforest_make(train_feat, test_feat):

    predictors = [f for f in test_feat.columns if f not in ['id','Blood_Sugar']]
    rf = linear_model.Ridge(alpha=0.05, normalize=True)

    num = 6
    train_preds = np.zeros(train_feat.shape[0])
    test_preds = np.zeros((test_feat.shape[0], num))
    kf = KFold(len(train_feat), n_folds=num, shuffle=True, random_state=0)

    for i,(train_index, test_index) in enumerate(kf):
        print('rf第{}次训练.....'.format(i+1))
        train_feat1 = train_feat.iloc[train_index]
        train_feat2 = train_feat.iloc[test_index]
        rf.fit(train_feat1[predictors], train_feat1['Blood_Sugar'])
        train_preds += rf.predict(train_feat[predictors])
        test_preds[:, i] = rf.predict(test_feat[predictors])

    train_preds = train_preds / num
   # train_preds[train_preds >= 7.8] = train_preds[train_preds >= 7.8] + (train_preds[train_preds >= 7.8] - 7.5) * 3
    print('rf线下得分：{}'.format(mean_squared_error(train_feat['Blood_Sugar'], train_preds)*0.5))

    test_preds = test_preds.mean(axis=1)

   # test_preds[test_preds >= 7.8] = test_preds[test_preds >= 7.8] + (test_preds[test_preds >= 7.8] - 7.5) * 3
    submission = pd.DataFrame({'pred': test_preds})
    submission.to_csv('../sub/sub_rf0.csv', index=False)

    return train_feat['Blood_Sugar'], train_preds / 1


if __name__ == '__main__':
    t1 = time.time()

    train_feat, test_feat = feature_extract()
    randomforest_make(train_feat, test_feat)

    t2 = time.time()

    print('rf用时{}秒'.format(t2 - t1))





