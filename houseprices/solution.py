import sys
import math
import pandas as pd
import numpy as np
import random as rnd
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier


def map_quality(dataset):
    """
    Map, fillna with zero and parse to int
    :param dataset:
    :return:
    """
    dataset = dataset.map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
    dataset = dataset.fillna(0)
    dataset = dataset.map(int)

    return dataset


def cut_values(df, count=5):
    """
    Split data by count and replace with iterations
    :param df:
    :param count:
    :return:
    """
    r = pd.cut(df, count).value_counts()
    replaceable = {}

    i = len(r)
    for key, item in r.items():
        replaceable[key] = i
        i -= 1

    return df.map(replaceable)


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

combine = [train_df, test_df]

for_dropping = [
    'Id', 'LotFrontage', 'Street', 'Alley', 'LandContour',
    'LandSlope', 'Condition1', 'Condition2', 'YearBuilt', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
    'BsmtUnfSF', 'LowQualFinSF', '1stFlrSF', '2ndFlrSF', 'MiscFeature', 'MoSold',
    'YrSold', 'PoolArea', 'Fence', 'BsmtCond', 'Electrical', 'GarageYrBlt', 'PavedDrive',
    'LotConfig', 'Heating', 'Foundation', 'MSSubClass', 'Utilities', 'PoolArea',
    'LowQualFinSF'
]

# print(train_df.info())
print(train_df['TotRmsAbvGrd'].value_counts())
# print(train_df['HouseStyle'].value_counts())
# print(train_df[['GarageYrBlt', 'SalePrice']].groupby('GarageYrBlt').mean().sort_values(by='SalePrice', ascending=False))
# sys.exit()

for dataset in combine:
    dataset.drop(for_dropping, axis=1, inplace=True)

    dataset['TotRmsAbvGrd'] = cut_values(dataset['TotRmsAbvGrd'], 4)
    # dataset.loc[[3, 2], 'TotRmsAbvGrd'] = 1
    # dataset.loc[[6, 7], 'TotRmsAbvGrd'] = 3
    # dataset.loc[[5, 8], 'TotRmsAbvGrd'] = 2
    # dataset.loc[dataset['TotRmsAbvGrd'] > 3, 'TotRmsAbvGrd'] = 1


    dataset['SaleType'] = dataset['SaleType'].map({'wd': 2, 'New': 1})
    dataset['SaleType'] = dataset['SaleType'].fillna(0)

    dataset['MSZoning'] = dataset['MSZoning'].map({
        'FV': 5, 'RL': 4, 'RH': 3, 'RM': 2, 'C (all)': 1
    })

    dataset['Porch'] = dataset['EnclosedPorch'] + dataset['OpenPorchSF'] + dataset['EnclosedPorch'] \
                       + dataset['3SsnPorch'] + dataset['ScreenPorch']

    dataset['PoolQC'] = map_quality(dataset['PoolQC'])
    dataset['PoolQC'] = dataset['PoolQC'].fillna(0)
    dataset.loc[dataset['PoolQC'] > 0, 'PoolQC'] = 0
    dataset.rename(columns={'PoolQC': 'hasPool'}, inplace=True)

    dataset['MSZoning'] = dataset['MSZoning'].fillna(5)
    dataset['MSZoning'] = dataset['MSZoning'].map(int)

    dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())
    dataset['GarageCars'] = dataset['GarageCars'].fillna(2)
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean())
    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0)
    dataset['FullBath'] = dataset['FullBath'].fillna(1)
    dataset['HalfBath'] = dataset['HalfBath'].fillna(1)

    dataset['GrLivArea'] = cut_values(dataset['GrLivArea'])
    dataset['MiscVal'] = cut_values(dataset['MiscVal'])
    dataset['MasVnrArea'] = cut_values(dataset['MasVnrArea'])
    dataset['TotalBsmtSF'] = cut_values(dataset['TotalBsmtSF'])
    dataset['WoodDeckSF'] = cut_values(dataset['WoodDeckSF'])

    dataset['BsmtBath'] = dataset['BsmtFullBath'] + dataset['BsmtHalfBath']
    dataset['Baths'] = dataset['FullBath'] + dataset['HalfBath']

    dataset['HouseStyle'] = dataset['HouseStyle'].map({
        '2.5fin': 70, '2Story': 60, '1Story': 50, 'SLvl': 40,
        '2.5Unf': 30, '1.5Fin': 30, 'SFoyer': 20, '1.5Unf': 10
    })

    dataset['BldgType'] = dataset['BldgType'].map({
        '1Fam': 3, 'TwnhsE': 3, 'Twnhs': 2, 'Duplex': 2, '2fmCon': 3
    })

    dataset['HouseType'] = dataset['HouseStyle'] + dataset['BldgType']
    dataset['HouseType'] = dataset['HouseType'].fillna(503)

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
    # dataset['Fireplaces'] = dataset['Fireplaces'] * dataset['FireplaceQu']
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

    dataset['LotShape'] = dataset['LotShape'].map({'Reg': 1, 'IR1': 0, 'IR2': 0, 'IR3': 0})

    dataset['GarageType'] = dataset['GarageType'].fillna(0)
    dataset['GarageType'] = dataset['GarageType'].map({
        'Attchd': 2, 'Detchd': 1, 'BuiltIn': 1, 'Basment': 1, 'CarPort': 1, '2Types': 1, 'NA': 0
    })
    dataset['GarageType'] = dataset['GarageType'].fillna(0)
    dataset['GarageType'] = dataset['GarageType'].map(int)

    dataset.loc[dataset['YearRemodAdd'] < 1962, 'YearRemodAdd'] = 1
    dataset.loc[(dataset['YearRemodAdd'] >= 1962) & (dataset['YearRemodAdd'] <= 1974), 'YearRemodAdd'] = 2
    dataset.loc[(dataset['YearRemodAdd'] > 1974) & (dataset['YearRemodAdd'] <= 1986), 'YearRemodAdd'] = 3
    dataset.loc[(dataset['YearRemodAdd'] > 1986) & (dataset['YearRemodAdd'] <= 1998), 'YearRemodAdd'] = 4
    dataset.loc[dataset['YearRemodAdd'] > 1998, 'YearRemodAdd'] = 5

    dataset.loc[dataset['LotArea'] <= 55000, 'LotArea'] = 1
    dataset.loc[(dataset['LotArea'] > 55000) & (dataset['LotArea'] <= 110000), 'LotArea'] = 2
    dataset.loc[(dataset['LotArea'] > 110000) & (dataset['LotArea'] <= 160000), 'LotArea'] = 3
    dataset.loc[dataset['LotArea'] > 160000, 'LotArea'] = 4

    dataset.loc[dataset['GarageArea'] <= 284, 'GarageArea'] = 3
    dataset.loc[(dataset['GarageArea'] > 284) & (dataset['GarageArea'] <= 567), 'GarageArea'] = 1
    dataset.loc[(dataset['GarageArea'] > 567) & (dataset['GarageArea'] <= 850), 'GarageArea'] = 2
    dataset.loc[(dataset['GarageArea'] > 850) & (dataset['GarageArea'] <= 1134), 'GarageArea'] = 4
    dataset.loc[dataset['GarageArea'] > 1134, 'GarageArea'] = 5

    dataset.drop([
        'EnclosedPorch', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'FireplaceQu', 'GarageCond', 'ExterCond', 'OverallCond', 'BldgType', 'HouseStyle',
        'BsmtHalfBath', 'BsmtFullBath', 'HalfBath', 'FullBath'
    ], axis=1, inplace=True)

print(test_df.columns.values)
# sys.exit()
X_train = train_df.drop("SalePrice", axis=1)
Y_train = train_df["SalePrice"]
X_test = test_df.copy()

# Applying different models
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(mean_squared_log_error(Y_train[1:], Y_pred))
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print(logreg.score(X_train, Y_train))
# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
# random_forest.score(X_train, Y_train)
# print(mean_squared_log_error(Y_train[1:], Y_pred))

#acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

