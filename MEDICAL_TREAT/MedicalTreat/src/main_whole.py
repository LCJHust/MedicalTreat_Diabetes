# -*- coding: utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
import _pickle as pickle
import matplotlib.pyplot as plt
import operator
import time

from feature_extract import get_features

def xgboost_train():

    trainData = get_features()
    label = trainData['Blood_Sugar']

    del trainData['id']
    del trainData['Blood_Sugar']

    feat = trainData.columns

    trainData,label = shuffle(trainData.values, label.values, random_state=2)

    trainNum = 5000                             #划分训练集和测试集
    trainData = trainData[:trainNum]
    trainLabel = label[:trainNum]
    testData = trainData[trainNum:]
    testLabel = label[trainNum:]

    trainData, validData, y_train, y_valid = train_test_split(trainData, trainLabel, test_size=0.2, random_state=2)
    dtrain = xgb.DMatrix(trainData, label=y_train)
    dtest = xgb.DMatrix(validData, label=y_train)

    param = {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 2,
             'min_child_weight': 4, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'reg:linear'}

    num_round = 350
    param['nthread'] = 8
    param['eval_metric'] = "rmse"
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    #bst = xgb.train(param, dtrain, num_round, evallist)
    bst = xgb.train(param, dtrain, num_round, evallist)
    pickle.dump(bst, open('xgboost', 'wb'))
    print("features num:", trainData.shape)

    features= [x for x in feat]
    ceate_feature_map(features)
    importance = bst.get_fscore(fmap='../cache/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance,columns=['feature','score'])
    df['score'] = df['score'] / df['score'].sum()
    df = df.sort_values('fscore', ascending=False)
    df.to_csv('../xgb_importance1.csv', index=False)

    features = [x for x in feat not in list(df['feature'])]
    print(features)

    df = df.sort_values('fscore',ascending=True)
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')  # 标题
    plt.xlabel('relative importance')  # x轴标签
    plt.show()

    bst = pickle.load(open('xgboost','rb'))
    dtest = xgb.DMatrix(trainData)
    pred = xgb.predict(dtest)
    result = pd.DataFrame(testLabel, columns='label')

    result['pred'] = pred

    final =  sum((result['pred'] - result['label']) * (result['pred'] - result['label'])) / (2 * len(result))
    print('final is:', final)

def ceate_feature_map(features):
    outfile = open('../cache/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

if __name__ == '__main__':

    t1 = time.time()

    xgboost_train()

    t2 = time.time()

    print('time:', t2-t1)







