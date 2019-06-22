import sys
import math
import pandas as pd
import numpy as np
import random as rnd
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier


def map_quality(dataset):
    # dataset = dataset.map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
    dataset = dataset.map({'Ex': 3, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 1})
    dataset = dataset.fillna(0)
    dataset = dataset.map(int)

    return dataset

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

combine = [train_df, test_df]

for_dropping = [
    'Id', 'LotFrontage', 'Street', 'Alley', 'LotShape', 'LandContour',
    'LandSlope', 'Condition1', 'Condition2', 'YearBuilt', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
    'BsmtUnfSF', 'LowQualFinSF', '1stFlrSF', '2ndFlrSF', 'MiscFeature', 'MoSold',
    'YrSold', 'PoolQC', 'Fence', 'BsmtCond', 'Electrical', 'GarageYrBlt', 'PavedDrive',
    'GarageType', 'LotConfig', 'YearRemodAdd', 'SaleType', 'Heating', 'Foundation', 'MSSubClass',
    'Utilities'
]

# print(train_df[['MSZoning', 'SalePrice']].groupby('MSZoning').mean().sort_values(by='SalePrice', ascending=False))
# sys.exit()

for dataset in combine:
    dataset.drop(for_dropping, axis=1, inplace=True)

    dataset['MSZoning'] = dataset['MSZoning'].map({
        'FV': 5, 'RL': 4, 'RH': 3, 'RM': 2, 'C (all)': 1
    })
    dataset['MSZoning'] = dataset['MSZoning'].fillna(2)
    dataset['MSZoning'] = dataset['MSZoning'].map(int)

    dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())
    dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].mean())
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean())
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].mean())
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].mean())

    dataset['HouseStyle'] = dataset['HouseStyle'].map({
        '2.5fin': 6, '2Story': 6, '1Story': 5, 'SLvl': 4,
        '2.5Unf': 3, '1.5Fin': 3, 'SFoyer': 2, '1.5Unf': 1
    })

    dataset['BldgType'] = dataset['BldgType'].map({
        '1Fam': 3, 'TwnhsE': 3, 'Twnhs': 2, 'Duplex': 2, '2fmCon': 3
    })

    dataset['HouseType'] = dataset['HouseStyle'] * dataset['BldgType']
    dataset['HouseType'] = dataset['HouseType'].fillna(15)

    dataset['Neighborhood'] = dataset['Neighborhood'].map({
        'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1,
        'BrkSide': 2, 'Edwards': 2, 'OldTown': 2,
        'Sawyer': 3, 'Blueste': 3,
        'SWISU': 4, 'NPkVill': 4, 'NAmes': 4, 'Mitchel': 4,
        'Mitchel': 5,
        'SawyerW': 6, 'NWAmes': 6, 'Gilbert': 6, 'Blmngtn': 6, 'CollgCr': 6,
        'Crawfor': 7, 'ClearCr': 7, 'Somerst': 7,
        'Veenker': 8, 'Timber': 8,
        'StoneBr': 9, 'NridgHt': 9, 'NoRidge': 9
    })

    dataset['BsmtQual'] = map_quality(dataset['BsmtQual'])
    dataset['FireplaceQu'] = map_quality(dataset['FireplaceQu'])
    dataset['Fireplaces'] = dataset['Fireplaces'] * dataset['FireplaceQu']
    dataset['Fireplaces'] = dataset['Fireplaces'].map(int)

    dataset['GarageQual'] = map_quality(dataset['GarageQual'])
    dataset['GarageCond'] = map_quality(dataset['GarageCond'])
    dataset['GarageQual'] = (dataset['GarageQual'] + dataset['GarageCond'])/2

    dataset['ExterQual'] = map_quality(dataset['ExterQual'])
    dataset['ExterCond'] = map_quality(dataset['ExterCond'])
    dataset['ExterQual'] = (dataset['ExterQual'] + dataset['ExterCond'])/2

    dataset['OverallQual'] = (dataset['OverallQual']/2 + dataset['OverallCond'])/2

    dataset['GarageFinish'] = dataset['GarageFinish'].map({'Fin': 2, 'RFn': 1, 'Unf': 1})
    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(0)
    dataset['GarageFinish'] = dataset['GarageFinish'].map(int)

    # dataset['Utilities'] = dataset['Utilities'].map({'AllPub': 1, 'NoSewr': 2, 'NoSewa': 1, 'ELO': 0})
    # dataset['Utilities'] = dataset['Utilities'].fillna(3)

    dataset['SaleCondition'] = dataset['SaleCondition'].map({
        'Abnormal': 0, 'Normal': 1, 'AdjLand': 0, 'Alloca': 0, 'Family': 0, 'Partial': 0
    })
    dataset['SaleCondition'] = dataset['SaleCondition'].fillna(0)
    dataset['SaleCondition'] = dataset['SaleCondition'].map(int)

    dataset['KitchenQual'] = map_quality(dataset['KitchenQual'])

    dataset['HeatingQC'] = map_quality(dataset['HeatingQC'])

    dataset['CentralAir'] = dataset['CentralAir'].map({'Y': 1, 'N': 0})

    dataset['Functional'] = dataset['Functional'].map({
        'Typ': 1, 'Min1': 2, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 4, 'Sev': 5, 'Sal': 5
    })
    dataset['Functional'] = dataset['Functional'].fillna(0)
    dataset['Functional'] = dataset['Functional'].map(int)

    # dataset.loc[dataset['LotArea'] <= 55000, 'LotArea'] = 1
    # dataset.loc[(dataset['LotArea'] > 55000) & (dataset['LotArea'] <= 110000), 'LotArea'] = 2
    # dataset.loc[(dataset['LotArea'] > 110000) & (dataset['LotArea'] <= 160000), 'LotArea'] = 3
    # dataset.loc[dataset['LotArea'] > 160000, 'LotArea'] = 4

    dataset.drop([
        'FireplaceQu', 'GarageCond', 'ExterCond', 'OverallCond', 'BldgType', 'HouseStyle'
    ], axis=1, inplace=True)


X_train = train_df.drop("SalePrice", axis=1)
Y_train = train_df["SalePrice"]
X_test = test_df.copy()

