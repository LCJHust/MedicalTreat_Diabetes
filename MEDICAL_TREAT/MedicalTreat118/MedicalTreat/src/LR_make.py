# -*-coding = utf-8 -*-
import numpy as np
import pandas as pd
import time
import operator
from pandas import Series, DataFrame
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn.cross_validation import cross_val_predict

from feature_extract import make_data_feat

train_feat, test_feat = make_data_feat()
lr = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
ridgereg = Ridge(alpha=0.00001, normalize=True)

### 未使用交叉验证

#del train_feat['id']
#del test_feat['id']

X_train = train_feat.drop('Blood_Sugar', axis=1)
X_train = X_train.drop('id', axis=1)
X_test = test_feat.drop('Blood_Sugar', axis=1)
X_test = X_test.drop('id', axis=1)

#predicted = cross_val_predict(lr, X_train, train_feat, cv=10)

print('X_train{}'.format(X_train.shape))
print('X_test{}'.format(X_test.shape))


lr.fit(X_train, train_feat['Blood_Sugar'])
ridgereg.fit(X_train, train_feat['Blood_Sugar'])

test_preds_lr = lr.predict(X_test)
train_preds_lr = lr.predict(X_train)

test_preds_ridge = ridgereg.predict(X_test)
train_preds_ridge = ridgereg.predict(X_train)

print(test_preds_lr.shape)
print(test_preds_ridge.shape)

#submission = pd.DataFrame({'pred': test_preds})
#submission.to_csv('../sub/sub_LR1.csv', index=False)
#feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
coef = pd.Series(lr.coef_, index = X_train.columns)
coef_abs = coef.abs().sort_values(ascending=False)
print(coef_abs)


print('线性模型：线下得分：   {}'.format(mean_squared_error(train_feat['Blood_Sugar'], train_preds_lr)*0.5))
print('线性L2正则化模型：线下得分：   {}'.format(mean_squared_error(train_feat['Blood_Sugar'], train_preds_ridge)*0.5))

