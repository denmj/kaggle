# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 100)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# train_df = train_df.drop(['Id'], axis=1)
# test_df = test_df.drop(['Id'], axis=1)

# Quick overlook on data
# print(train_df.info())
# print(test_df.info())
# print('_'*40)
# print(train_df.describe(include='all'))
# print('_'*40)
# print(train_df.head())
# print(test_df.head())
# print(train_df.shape)
# print(test_df.shape)

#
# print(train_df[['MSSubClass', 'SalePrice']].groupby(['MSSubClass']).mean().sort_values(by='SalePrice', ascending=False))
# print(train_df[['HouseStyle', 'SalePrice']].groupby(['HouseStyle']).mean().sort_values(by='SalePrice', ascending=False))
# print(train_df[['BldgType', 'SalePrice']].groupby(['BldgType']).mean().sort_values(by='SalePrice', ascending=False))

# Серый во глянь на дату и анализ
# Lets check different House/Building styles by prices and over whole period of time
# As we can see YtY prices stay relatively flat
#  MSSubClass contains same duplicate info as HouseStyle/BldgType , therefore we can eventually drop it from data set
# Both columns are categorical, Columns doesnt contain NaN's values
# Columns HouseStyle/BldgType we can turn into ordinal
# HouseStyle}
# sns.barplot(x="YrSold", y="SalePrice", hue="HouseStyle", data=train_df.sort_values(by='SalePrice', ascending=False))
# plt.show()
# sns.barplot(x="YrSold", y="SalePrice", hue="BldgType", data=train_df.sort_values(by='SalePrice', ascending=False))
# plt.show()
# sns.barplot(x="YrSold", y="SalePrice", hue="MSSubClass", data=train_df.sort_values(by='SalePrice', ascending=False))
# plt.show()

both_data = [train_df, test_df]

housestyle_mapping = {"SFoyer": 1, "SLvl": 1, "1Story": 2, "1.5Fin": 3, "1.5Unf": 3,
                      "2Story": 4, "2.5Fin": 5, "2.5Unf": 5}
bldgtype_mapping = {"2fmCon": 1, "Duplex": 1, "Twnhs": 2, "TwnhsE": 2, "1Fam": 3}
zoning_mapping = {"C (all)": 1, "RM": 2, "RH": 3, "RL": 4, "FV": 5}

for dataset in both_data:
    dataset['HouseStyle'] = dataset['HouseStyle'].map(housestyle_mapping)
    dataset['BldgType'] = dataset['BldgType'].map(bldgtype_mapping)
    dataset['MSZoning'] = dataset['MSZoning'].map(zoning_mapping)

for dataset in both_data:
    dataset['HouseBuiltComb'] = dataset['HouseStyle'] + dataset['BldgType']

for dataset in both_data:
    dataset.loc[dataset['LotArea'] <= 55000, 'LotArea'] = 1
    dataset.loc[(dataset['LotArea'] > 55000) & (dataset['LotArea'] <= 110000), 'LotArea'] = 2
    dataset.loc[(dataset['LotArea'] > 110000) & (dataset['LotArea'] <= 160000), 'LotArea'] = 3
    dataset.loc[dataset['LotArea'] > 160000, 'LotArea'] = 4

train_df = train_df.drop(['HouseStyle', 'BldgType', 'MSSubClass', 'LotFrontage'], axis=1)
test_df = test_df.drop(['HouseStyle', 'BldgType', 'MSSubClass', 'LotFrontage'], axis=1)

print(train_df[['LotArea', 'SalePrice']].groupby(['LotArea']).mean().sort_values(by='LotArea', ascending=False))

print(train_df.shape)
print(test_df.shape)

# print(train_df[['HouseBuiltComb', 'SalePrice']])
# print(train_df[['MSZoning']])
# print(train_df[['MSZoning', 'SalePrice']].groupby(['MSZoning']).mean().sort_values(by='SalePrice', ascending=False))
