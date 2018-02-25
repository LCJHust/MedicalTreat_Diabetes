# -*- coding: utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb  #导入xgboost模块
import _pickle as pickle
import matplotlib.pyplot as plt
import operator
import time

from feature_extract import get_features      #从feature_extract.py文件中获取get_feature()函数

def xgboost_train():  # xgboost训练及线下测评

    trainData = get_features()  #提取特征,删除了性别不明的样本，将性别和年龄都转换成0/1值
    label = trainData[['Blood_Sugar']]  # 获得标签

    del trainData['id']    #删除id一列
    del trainData['Blood_Sugar']      #删除血糖值一列

    feat = trainData.columns    #列标签

    trainData, label = shuffle(trainData.values, label.values, random_state=2)  # 打乱顺序
    trainNum = 5400  # 训练数据个数
    testData = trainData[trainNum:5641]    #测试数据5000-5641
    testLabel = label[trainNum:5641]
    trainData = trainData[0:trainNum]      #训练数据1-5000
    label = label[0:trainNum]

    trainData, X_test, y_train, y_test = train_test_split(trainData, label, test_size=0.1, random_state=2)  #按0.2的比例抽取训练集和测试集
    dtrain = xgb.DMatrix(trainData, label=y_train)   #加载数据
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 2,
             'min_child_weight': 4, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'reg:linear'}

    # learning_rate/eta:学习速率，更新叶子节点权重时，乘以该系数，避免步长过大
    # n_estimators:
    # max_depth:树的最大深度
    # min_child_weight：叶子几点啊实际需要的权重的最小总和。在线性模型下，是指建立每个模型所需要的最小样本数。调大这个参数能够控制过拟合。
    # gamma：在树的叶节点上进行进一步分区所需的最小损失减少。 算法越大，越保守。后剪枝时，用于控制是否后剪枝的参数。
    # subsample：用于训练的子样本占整个样本集合的比例
    # colsample_bytree：在建立树时对特征采样的比例
    # scale_pos_weight：控制正负权重的平衡，对不平衡类有用。 一个典型的值要考虑：sum（负数）/ sum（正数）
    # silent：0表示打印运行时的信息，1表示以缄默方式运行
    # objective：定义最小化损失函数的类型

    num_round = 350    #迭代次数  ?
    param['nthread'] = 8   #用于运行xgboost的并行线程数
    param['eval_metric'] = "rmse"   #校验数据所用评价指标，rmsd均方根误差
    evallist = [(dtest, 'eval'), (dtrain, 'train')]    #测试误差，训练误差
    bst = xgb.train(param, dtrain, num_round, evallist)   #训练模型
    pickle.dump(bst, open('xgboost', 'wb'))   #对象将其当前状态写入到临时或永久性存储区
    print("features num:", (trainData.shape))  #打印训练   features num: (4000, 48)

    features = [x for x in feat]
    ceate_feature_map(features)     #？？？
    importance = bst.get_fscore(fmap='../cache/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df = df.sort_values('fscore', ascending=False)        #按照fscore降序排列
    df.to_csv("../xgb_importance.csv", index=False)

    features = [x for x in feat if x not in list(df['feature'])]
    print(features)  # ['HbeAg', 'age_0', 'age_3', 'age_7']

    df = df.sort_values('fscore', ascending=True)    #按照fscore升序排列  ？？？
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10)) #水平柱状图，没有图例，图像大小
    plt.title('XGBoost Feature Importance')  #标题
    plt.xlabel('relative importance')   #x轴标签
    plt.show()

    bst = pickle.load(open('xgboost', 'rb'))  # 线下测评 反序列化
    dtest = xgb.DMatrix(testData)
    pred = bst.predict(dtest)   #预测结果
    result = pd.DataFrame(testLabel, columns=['label'])

    result['pred'] = pred


    final = sum((result['pred'] - result['label']) * (result['pred'] - result['label'])) / (2 * len(result))
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

    print('time:', t2 - t1)




