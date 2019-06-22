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

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 100)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# train_df = train_df.drop(['Id'], axis=1)
# test_df = test_df.drop(['Id'], axis=1)

# Quick overlook on data
# print(train_df.info())

print(train_df.info())
# print('_'*40)
# print(train_df.describe(include='all'))
# print('_'*40)
# print(train_df.head())
# print(test_df.head())
# print(train_df.shape)
# print(test_df.shape)

# ----------------Data exploration and visualization

# Lets see Target Data
# Data skewed to cheaper prices with long tails in expensive price range
# sale_price_df = pd.DataFrame({'SalesP Hist': train_df.SalePrice, 'SaleP LogHist': np.log(train_df.SalePrice)})
# sale_price_df.hist(histtype='bar', grid=False, bins=25)
# plt.show()

# Checking MSSubClass - Categorical,
# Distribution - '20' - 36% , '60' - 20% , '50' - 9% , '120' - 5% rest is less than 5%
#
# print(train_df['MSSubClass'].value_counts(normalize=True))
# sns.countplot(x='MSSubClass', data=train_df, color='blue')
# plt.show()

# Checking Zoning - Categorical
# Distribution - 92% data in RL(78%) and RM(14%)
#
# print(train_df['MSZoning'].value_counts(normalize=True))
# sns.countplot(x='MSZoning', data=train_df, color='blue')

# plt.show()

# Checking LotFrontage - Numerical data
# Missing - 259 (or not available option for some houses)
# print(train_df['LotFrontage'].describe())
# print(train_df['LotFrontage'].isna().sum())
# print(train_df['LotFrontage'].isnull().sum())

# Checking LotArea - Numerical data
# LotArea / SalePrice correlation ?
# Good correlation between sale price and lot area
#
# print(train_df['LotArea'].describe())
# plt.gca().set_xlim([0,25000])
# plt.plot(train_df.LotArea, np.log(train_df.SalePrice), '.', alpha=0.3)
# plt.show()

# Checking Street - Categorical data
# Almost 100% Streets are Pavement, only 6 out of 1454 Gravel
# If all houses have Pave streets then Sale PRice won't depend on this feature
# print(train_df['Street'].value_counts())

# Checking Alley - Categorical data
# 1369 missing entries - this potentially could be dropped
#
# print(train_df['Alley'].value_counts())
# print(train_df['Alley'].isna().sum())

# LotShape - Categorical data
# make 2 cats - Reg and Irregular
# print(train_df['LotShape'])
# print(train_df['LotShape'].value_counts())
# print(train_df['Alley'].isnull().sum())

# LandContour - Cat Data
# ~90% lv(Flat surface) , Could be grouped by Level and Rest
# print(train_df['LandContour'])
# print(train_df['LandContour'].value_counts())
# print(train_df[['LandContour', 'SalePrice']].groupby('LandContour').mean())

# Utilities - cat data
# Looks like all houses got All utility options, so this data won't be a good predictor
# Could be dropped
# print(train_df['Utilities'])
# print(train_df['Utilities'].value_counts())
#
# LotConfig - Cat data
# Group by 4, join FR2-FR3 - turn to ordinal
# print(train_df['LotConfig'])
# print(train_df['LotConfig'].value_counts())
# print(train_df[['LotConfig', 'SalePrice']].groupby('LotConfig').mean())

# LandSlope  Cat data
# 90% of it Normal slope land , rest of 2 cat put into one cat : Abnormal slope
# make ordinal
# print(train_df['LandSlope'])
# print(train_df['LandSlope'].value_counts())
# print(train_df[['LandSlope', 'SalePrice']].groupby('LandSlope').mean())

# Neighborhood - Cat data
# I would assume this feature will have strong correlation with SalePrice
# One of the Main predictors along with LotArea
# print(train_df['Neighborhood'])
# print(train_df['Neighborhood'].value_counts())
# print(train_df[['Neighborhood', 'SalePrice']].groupby('Neighborhood').mean().sort_values(by='SalePrice', ascending=False))
#
# sns.catplot(x="Neighborhood", y="SalePrice", hue="Neighborhood", kind="swarm",height=7, aspect=5, data=train_df);
# plt.show()

# Condition1 - Cat Data
# Again large majority (~90%)of house located in nornal proximity to various conditions
# rest 10% are Not Normal and Good
# Could be grouped onto 3 cats or 2
# print(train_df['Condition1'])
# print(train_df['Condition1'].value_counts())
# print(train_df[['Condition1', 'SalePrice']].groupby('Condition1').mean())
#
# Condition2 is duplicate
# These two columns could be joined and presented as an artificial feature for a model
# If Con1 Bad but Con2 Good = Con12 Norm


# BldgType and HouseStype  could be dropped as both cols represented in MSSubClass

# OverallQual and OverllCond have a strong correlation with SalePrice
# Another 2 important predictors for our model
# print(train_df['OverallQual'].value_counts())
# print(train_df[['OverallQual', 'SalePrice']].groupby('OverallQual').mean())
# sns.catplot(x="OverallQual", y="SalePrice", hue="OverallQual", kind="swarm",height=7, aspect=5, data=train_df);
# plt.show()

#
# print(train_df['OverallCond'].value_counts())
# print(train_df[['OverallCond', 'SalePrice']].groupby('OverallCond').mean())
# sns.catplot(x="OverallCond", y="SalePrice", hue="OverallCond", kind="swarm",height=7, aspect=5, data=train_df);
# plt.show()


# YearBuilt - log of SalePrice , show a correlation that house prices were going up over years
# Good predictor
# print(train_df['YearBuilt'].value_counts())
# print(train_df[['YearBuilt', 'SalePrice']].groupby('YearBuilt').mean())
# plt.plot(train_df.YearBuilt, np.log(train_df.SalePrice), '.', alpha=0.3)
# plt.show()








# Combine both sets of data set for modification convenience
both_data = [train_df, test_df]


# Turn HouseStyle into ordinal
# housestyle_mapping = {"SFoyer": 1, "SLvl": 1, "1Story": 2, "1.5Fin": 3, "1.5Unf": 3,
#                       "2Story": 4, "2.5Fin": 5, "2.5Unf": 5}
# # Turn  BldgType to ordinal
# bldgtype_mapping = {"2fmCon": 1, "Duplex": 1, "Twnhs": 2, "TwnhsE": 2, "1Fam": 3}
#
# # Turn MSZoning to ordinal
# zoning_mapping = {"C (all)": 1, "RM": 2, "RH": 3, "RL": 4, "FV": 5}
#
# for dataset in both_data:
#     dataset['HouseStyle'] = dataset['HouseStyle'].map(housestyle_mapping)
#     dataset['BldgType'] = dataset['BldgType'].map(bldgtype_mapping)
#     dataset['MSZoning'] = dataset['MSZoning'].map(zoning_mapping)
#
# # Creating new feature HouseStyleType
# for dataset in both_data:
#     dataset['HouseStyleType'] = dataset['HouseStyle'] + dataset['BldgType']

# for dataset in both_data:
#     dataset.loc[dataset['LotArea'] <= 55000, 'LotArea'] = 1
#     dataset.loc[(dataset['LotArea'] > 55000) & (dataset['LotArea'] <= 110000), 'LotArea'] = 2
#     dataset.loc[(dataset['LotArea'] > 110000) & (dataset['LotArea'] <= 160000), 'LotArea'] = 3
#     dataset.loc[dataset['LotArea'] > 160000, 'LotArea'] = 4

# All dropped columns
# train_df = train_df.drop(['HouseStyle', 'BldgType', 'MSSubClass', 'LotFrontage'], axis=1)
# test_df = test_df.drop(['HouseStyle', 'BldgType', 'MSSubClass', 'LotFrontage'], axis=1)


# print(train_df[['LotArea', 'SalePrice']].groupby(['LotArea']).mean().sort_values(by='LotArea', ascending=False))
# print(train_df[['HouseStyleType', 'SalePrice']].groupby(['HouseStyleType']).mean().sort_values(by='HouseStyleType', ascending=False))
# print(train_df[['MSZoning', 'SalePrice']].groupby(['MSZoning']).mean().sort_values(by='MSZoning', ascending=False))

# plt.hist(train_df.SalePrice, bins=20)
# plt.show()
# plt.hist(np.log(train_df.SalePrice), bins=20)
# plt.show()
#
# sns.countplot(train_df.MSZoning)
# plt.show()

#
# b = plt
# ax1 = b.gca()
# ax1.set_xlim([0,25000])
# b.plot(train_df.LotArea, np.log(train_df.SalePrice), '.', alpha=0.3)
# b.show()


#print(train_df[['Alley']])


# print(train_df.shape)
# print(test_df.shape)

# print(train_df[['MSZoning', 'SalePrice']].groupby(['MSZoning']).mean().sort_values(by='SalePrice', ascending=False))
