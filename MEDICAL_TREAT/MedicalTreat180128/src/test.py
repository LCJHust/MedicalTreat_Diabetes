# coding:utf-8

# 引入需要的包

import numpy as np
import pandas as pd

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')


import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

# 读取数据

train = pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
test = pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')

"""
['id', '性别', '年龄', '体检日期', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例',
'甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体',
'乙肝核心抗体', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数',
'血小板平均体积', '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%', '血糖']
"""
columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT',
                  'MPV', 'PDW','PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']

train.columns = columns_rename
columns_rename.pop()
test.columns = columns_rename

print('train shape',train.shape)
print('test shape',test.shape)

train_ID = train['id']
test_ID = test['id']

print('train feature shape',train.shape)
print('test feature shape',test.shape)

# 查看数据
print(train.head())
print(test.head())

# 查看特征列
print(train.columns)
data = pd.concat([train,test],axis=0)
print(data.isnull().sum()/len(data))

# #特征相关性
# from pylab import mpl
# from scipy.special import boxcox1p
# lam = 0.15
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# t_f = train['TG']
# fig ,ax = plt.subplots()
# # tmp,lambda_  = stats.boxcox(train['Blood_Sugar'])
# ax.scatter(x = t_f,y=train['Blood_Sugar'])
# plt.ylabel('Blood_Sugar')
# plt.xlabel('TG')
# plt.show()


# 血糖 is the variable we need to predict. So let's do some analysis on this variable first.
def pred_distribute():
    train2 = pd.read_csv('../sub/sub_final001.csv',encoding='gbk')
    sns.distplot(train2['pred'],fit=norm)
    (mu,sigma) = norm.fit(train2['pred'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('pred')

    fig = plt.figure()
    res = stats.probplot(train2['pred'], plot=plt)
    plt.show()


# sns.distplot(train['Blood_Sugar'],fit=norm)
#
# (mu,sigma) = norm.fit(train['Blood_Sugar'])
#
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('Blood_Spread')
#
# fig = plt.figure()
# res = stats.probplot(train['Blood_Sugar'], plot=plt)
# plt.show()

# ntrain = train.shape[0]
# ntest = test.shape[0]
# y_train = train['Blood_Sugar'].values
# all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.drop(['Blood_Sugar'], axis=1, inplace=True)
# print("all_data size is : {}".format(all_data.shape))
#
# all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
# all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
# missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
#
# corrmat = train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,vmax=0.9,square=True)


if __name__ == '__main__':
    pred_distribute( )