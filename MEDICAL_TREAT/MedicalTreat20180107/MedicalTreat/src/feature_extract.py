# -*- coding: utf-8 -*-
import pandas as pd


train_data_file = '../data/d_train_20180102.csv'
test_data_file = '../data/d_test_A_20180102.csv'

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                  'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']

mean = None

def convert_age(age):
    if age < 20:
        return 0
    elif age > 20 and age <= 30:
        return 1
    elif age > 30 and age <= 40:
        return 2
    elif age > 40 and age <= 50:
        return 3
    elif age > 50 and age <= 60:
        return 4
    elif age > 60 and age <= 70:
        return 5
    elif age > 80:
        return 6
    else:
        return -1

def read_train_file():
    train_data = pd.read_csv(train_data_file, encoding="gb2312")
    train_data.columns = columns_rename
    train_data = train_data[(train_data['Sex'] == u'男') | (train_data['Sex'] == u'女')]
    train_data['Age'] = train_data['Age'].map(convert_age)
    age_df = pd.get_dummies(train_data['Age'], prefix="age")
    sex_df = pd.get_dummies(train_data['Sex'], prefix="sex")
    sex_df.columns = ['sex_m', 'sex_w']
    del train_data['Age']
    del train_data['Sex']
    del train_data['Date']
    train_data = pd.concat([train_data, age_df, sex_df], axis=1)
    train_data_m = train_data[train_data['sex_m']==1]
    train_data_w = train_data[train_data['sex_w']==1]
    #global mean_w
   # global mean_m
    mean_w = train_data_w.mean()
    mean_m = train_data_m.mean()
    train_data_w = train_data_w.fillna(mean_w)
    train_data_m = train_data_w.fillna(mean_m)
    train_data = pd.concat([train_data_w, train_data_m])
    return train_data


def make_train_data():
    feat = read_train_file()
#    print(feat)
    return feat




def read_test_file():
    test_data = pd.read_csv(test_data_file, encoding="gb2312")
    columns_rename.pop()   #取出栈顶元素，'Blood_Sugar'，因为测试数据没有血糖值
    test_data.columns = columns_rename
    test_data = test_data[(test_data['Sex'] == u'男') | (test_data['Sex'] == u'女')]
    test_data['Age'] = test_data['Age'].map(convert_age)
    age_df = pd.get_dummies(test_data['Age'], prefix="age")
    sex_df = pd.get_dummies(test_data['Sex'], prefix="sex")
    del test_data['Age']
    del test_data['Sex']
    del test_data['Date']
    test_data = pd.concat([test_data, age_df, sex_df], axis=1)
  #  test_data = test_data.fillna(mean)
    return test_data


def make_test_data():
    feat = read_test_file()
    return feat



if __name__ == '__main__':

    # read_train_file()
    read_test_file()