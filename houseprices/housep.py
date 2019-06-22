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

pd.set_option('display.max_rows', 1460)
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 100)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df = train_df.drop(['Id'], axis=1)
test_df = test_df.drop(['Id'], axis=1)

both_data = [train_df, test_df]

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

#
# print(train_df['MSSubClass'].value_counts(normalize=True))
# sns.countplot(x='MSSubClass', data=train_df, color='blue')
# plt.show()


# Checking Zoning - Categorical
# Distribution - 92% data in RL(78%) and RM(14%)

# print(train_df['MSZoning'].value_counts(normalize=True))
# sns.countplot(x='MSZoning', data=train_df, color='blue')
# plt.show()

# MSZoning
for dataset in both_data:
    dataset['MSZoning'] = dataset['MSZoning'].replace(['RL', 'RM', 'FV', 'RH'], 'Residential')
    dataset['MSZoning'] = dataset['MSZoning'].replace(['C (all)'], 'Commercial')

zoning_mapping = {"Commercial": 0, "Residential": 1}
for dataset in both_data:
    dataset['MSZoning'] = dataset['MSZoning'].map(zoning_mapping)

# Checking LotFrontage - Numerical data
# Missing - 259 (or not available option for some houses)

for dataset in both_data:
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(0)
print(train_df['LotFrontage'].describe())
train_df['LotFrontageB'] = pd.cut(train_df['LotFrontage'], 5)

pd.DataFrame({'LotFrontage': train_df.LotFrontage}).hist(histtype='bar', grid=False, bins=25)

plt.show()


# for dataset in both_data:
#     dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
#     dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

# Checking LotArea - Numerical data
# LotArea / SalePrice correlation ?
# Good correlation between sale price and lot area

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
# # Good predictor
# print(train_df['YearBuilt'].value_counts())
# print(train_df[['YearBuilt', 'SalePrice']].groupby('YearBuilt').mean())
# print(train_df[['YearBuilt', 'SalePrice']].groupby('YearBuilt').mean())
# plt.plot(train_df.YearBuilt, np.log(train_df.SalePrice), '.', alpha=0.3)
# plt.show()
# #

# YearRemodAdd - numerical ord , comparing YB and YRA we can see affected houses between 1880 and 1950
# ANy changes in YBD could be mapped to YB
# print(train_df['YearRemodAdd'].value_counts())
# print(train_df[['YearRemodAdd', 'SalePrice']].groupby('YearRemodAdd').mean())
# print(pd.pivot_table(train_df[['YearRemodAdd', 'YearBuilt', 'SalePrice']], values='SalePrice', index=['YearRemodAdd', 'YearBuilt'],))
# print(train_df[['YearRemodAdd', 'YearBuilt' ,'SalePrice']].groupby('YearBuilt')).mean()
# plt.plot(train_df.YearRemodAdd, np.log(train_df.SalePrice), '.', alpha=0.3)
# plt.show()
#

# Cat - Data turn into ord / 3 grups
# print(train_df['RoofStyle'])
# print(train_df['RoofStyle'].value_counts())
# print(train_df[['RoofStyle', 'SalePrice']].groupby('RoofStyle').mean())

# All Roofls are made of almost 1 material / not a good predictor
# print(train_df['RoofMatl'])
# print(train_df['RoofMatl'].value_counts())
# print(train_df[['RoofMatl', 'SalePrice']].groupby('RoofMatl').mean())

# Exterior1st / Exterior1st
# Do same as Con1 and Cond2
# print(train_df['Exterior1st'])
# print(train_df['Exterior1st'].value_counts())
# print(train_df[['Exterior1st', 'SalePrice']].groupby('Exterior1st').mean())
# print(train_df['Exterior1st'])
# print(train_df['Exterior1st'].value_counts())
# print(train_df[['Exterior1st', 'SalePrice']].groupby('Exterior1st').mean())


# MasVnrType /MasVnrArea
# 2 goups (No type + crkcmn  / BrkFace + Stone)
#
# print(train_df['MasVnrType'])
# print(train_df['MasVnrType'].value_counts())
# print(train_df[['MasVnrType', 'SalePrice']].groupby('MasVnrType').mean())
# print(train_df['MasVnrArea'])
# print(train_df['MasVnrArea'].value_counts())
# print(train_df[['MasVnrArea', 'SalePrice']].groupby('MasVnrArea').mean())
# print(pd.pivot_table(train_df[['MasVnrType', 'MasVnrArea', 'SalePrice']], values='SalePrice', index=['MasVnrType', 'MasVnrArea']))
#

# ExterQual, ExterCond - Turn ordinal  , poors and excellent difference from good and fair signufucant
# we dont join and just turn it into ordinal

# Foundation - turn to ordinal (full set)


# All categorical basement data could be joined and new Basement column created
# gaps must be filled
# Leave total basement area as numerical but make it ordinal
# print(train_df['BsmtQual'])
# print(train_df['BsmtQual'].value_counts())
# print(train_df[['BsmtQual', 'SalePrice']].groupby('BsmtQual').mean())
# print(train_df['BsmtCond'])
# print(train_df['BsmtCond'].value_counts())
# print(train_df[['BsmtCond', 'SalePrice']].groupby('BsmtCond').mean())
#
# Use GrLivArea - its a comb of 1st and 2nd
# Think about what to do with Lower Q Finish of the floor
# print(train_df[['1stFlrSF', '2ndFlrSF', 'GrLivArea']])
# print(train_df[['GrLivArea', 'SalePrice']].groupby('GrLivArea').mean())


# combine 0 - 0  half is 1 and ful is 2 , new column and drop 2 old
# print(train_df[['BsmtFullBath', 'BsmtHalfBath']])
#
#
# Leave as it is
# print(train_df['FullBath'].value_counts())
# print(train_df['HalfBath'].value_counts())
# print(train_df[['FullBath', 'HalfBath']])
#


# #Functional
# print(train_df['Functional'].value_counts())
# print(train_df[['Functional', 'SalePrice']].groupby('Functional').mean())

# F irePlace - Turn into ordinal
# print(train_df['Fireplaces'].value_counts())
#
# print(train_df['FireplaceQu'].value_counts())

# Drop PoolQC , MiscF, Fence, MiscF


# Combine both sets of data set for modification convenience


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


# print(train_df[['Alley']])


# print(train_df.shape)
# print(test_df.shape)

# print(train_df[['MSZoning', 'SalePrice']].groupby(['MSZoning']).mean().sort_values(by='SalePrice', ascending=False))
