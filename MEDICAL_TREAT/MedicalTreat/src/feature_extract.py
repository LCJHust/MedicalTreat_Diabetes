# -*- coding: utf-8 -*-
import pandas as pd

train_data_file = '../data/d_train_20180102.csv'      #读取训练数据

columns_rename = ['id', 'Sex', 'Age', 'Date', 'AST', 'ALT', 'ALP', 'GGT', 'TP', 'ALB', 'GLB', 'AG',
                  'TG', 'TC', 'HDL_C', 'LDL_C', 'Urea', 'Cre', 'UA', 'HBsAg' ,'HBsAb', 'HbeAg', 'HBeAb',
                  'HBcAb', 'WBC' ,'RBC' , 'HGB', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW',
                  'PCT', 'Neutrophil', 'Lymph', 'Monocytes', 'Acidophilic' ,'Basophil' ,'Blood_Sugar']     #列表，每一列重新命名

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
    elif age > 80 and age <= 90:
        return 6
    elif age > 90:
        return 7
    else:
        return -1

def read_train_file():          #函数：读取训练数据，并进行适当的转换
    train_data = pd.read_csv(train_data_file, encoding="gb2312")  #读入一个DataFrame，encoding='gb23122'读取中文
    train_data.columns = columns_rename   #列索引为columns_rename，将中文替换成英文
    train_data = train_data[(train_data['Sex'] == u'男') | (train_data['Sex'] == u'女')]  #将性别为男或者性别为女的样本留下，性别不明的样本剔除
    train_data['Age'] = train_data['Age'].map(convert_age)   #map()将一个自定义函数应用于Series结构中的每一个元素
    age_df = pd.get_dummies(train_data["Age"], prefix="age")
    sex_df = pd.get_dummies(train_data["Sex"], prefix="sex")   #将不同的值转换为0/1
    sex_df.columns = ['sex_m', 'sex_w']
    del train_data['Age']
    del train_data['Sex']
    del train_data['Date']
    train_data = pd.concat([train_data, age_df, sex_df], axis=1)

    train_data = train_data.fillna(0)        #将缺失值填充为0
    return train_data


def get_features():
    feat = read_train_file()   #删除了性别不明的样本，将性别和年龄都转换成0/1值
    return feat


if __name__ == '__main__':

    read_train_file()