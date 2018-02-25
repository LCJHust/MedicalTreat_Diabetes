# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
import _pickle as pickle
import matplotlib.pyplot as plt
import operator
import time

from feature_extract import make_train_data, make_test_data

def xgboost_train():  # xgboost训练及线下测评

    trainData = make_train_data()  # 提取特征
    label = trainData['Blood_Sugar']  # 获得标签

    del trainData['id']
    del trainData['Blood_Sugar']

    feat = trainData.columns

    trainData, label = shuffle(trainData.values, label.values, random_state=3)  # 打乱顺序
    trainNum = 5000  # 训练数据个数
    testData = trainData[trainNum:5641]
    testLabel = label[trainNum:5641]
    trainData = trainData[0:trainNum]
    label = label[0:trainNum]

    trainData, X_test, y_train, y_test = train_test_split(trainData, label, test_size=0.001, random_state=2)
    dtrain = xgb.DMatrix(trainData, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 2,
             'min_child_weight': 4, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'reg:linear'}

    num_round = 350
    param['nthread'] = 8
    param['eval_metric'] = "rmse"
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    pickle.dump(bst, open('xgboost', 'wb'))
    print("features num:", (trainData.shape))

    features = [x for x in feat]
    ceate_feature_map(features)
    importance = bst.get_fscore(fmap='../cache/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df = df.sort_values('fscore', ascending=False)
    df.to_csv("../xgb_importance.csv", index=False)

    features = [x for x in feat if x not in list(df['feature'])]
    print (features)

    df = df.sort_values('fscore', ascending=True)
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.show()

    bst = pickle.load(open('xgboost', 'rb'))  # 线下测评
    dtest = xgb.DMatrix(testData)
    pred = bst.predict(dtest)
    result = pd.DataFrame(testLabel, columns=['label'])
    result['pred'] = pred

    final = sum((result['pred'] - result['label']) * (result['pred'] - result['label'])) / (2 * len(result))
    print(final)


def ceate_feature_map(features):
    outfile = open('../cache/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def xgboost_sub():
    testData = make_test_data()
    result = testData[['id']]
    del testData['id']
    bst = pickle.load(open('xgboost', 'rb'))
    testData = xgb.DMatrix(testData.values)
    pred = bst.predict(testData)
    result['pred'] = pred
    result = result.sort_values('id')
    result = result[['pred']]
    result.to_csv('../sub/sub0.csv', index=False, header=False)


if __name__ == '__main__':

    t1 = time.time()

    xgboost_train()
    xgboost_sub()

    t2 = time.time()

    print ('time:', t2 - t1)




