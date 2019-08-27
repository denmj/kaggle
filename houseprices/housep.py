# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


pd.set_option('display.max_rows', 1460)
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 100)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

both_data = [train_df, test_df]

# train_df = train_df.drop(['Id'], axis=1)
# test_df = test_df.drop(['Id'], axis=1)

# Quick overlook on data
# print(train_df.info())

# print(train_df.info())
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

# print(train_df['MSZoning'].value_counts(normalize=True))
# sns.countplot(x='MSZoning', data=train_df, color='blue')
# plt.show()

zoning_mapping = {"Commercial": 0, "Residential": 1}
lot_shape_mapping = {"Irregular": 0, "Reg": 1}
land_cnt_mapping = {"Rest": 0, "Lvl": 1}
lot_cnfg_mapping = {"FR2-3": 0, "CulDSac": 1, "Corner": 2, "Inside": 3}
land_slope_mapping = {"AbnormalSlope": 0, "NormalSlope": 1}
cond1_mapping = {"Norm": 2, "Bad": 1, "Good": 3}
cond2_mapping = {"Norm": 2, "Bad": 1, "Good": 3}

for dataset in both_data:


    dataset['MSZoning'] = dataset['MSZoning'].replace(['RL', 'RM', 'FV', 'RH'], 'Residential')
    dataset['MSZoning'] = dataset['MSZoning'].replace(['C (all)'], 'Commercial')


    dataset['LotShape'] = dataset['LotShape'].replace(['IR1', 'IR2', 'IR3'], 'Irregular')
    dataset['LandContour'] = dataset['LandContour'].replace(['Bnk', 'HLS', 'Low'], 'Rest')
    dataset['LotConfig'] = dataset['LotConfig'].replace(['FR2', 'FR3'], 'FR2-3')
    dataset['LandSlope'] = dataset['LandSlope'].replace(['Gtl'], 'NormalSlope')
    dataset['LandSlope'] = dataset['LandSlope'].replace(['Mod', 'Sev'], 'AbnormalSlope')
    dataset['Condition1'] = dataset['Condition1'].replace(['Artery', 'Feedr', 'RRAe', 'RRAn', 'RRNn'], 'Bad')
    dataset['Condition1'] = dataset['Condition1'].replace(['PosA', 'PosN'], 'Good')
    dataset['Condition2'] = dataset['Condition2'].replace(['Artery', 'Feedr', 'RRAe', 'RRAn', 'RRNn'], 'Bad')
    dataset['Condition2'] = dataset['Condition2'].replace(['PosA', 'PosN'], 'Good')

    dataset['LotShape'] = dataset['LotShape'].map(lot_shape_mapping)
    dataset['MSZoning'] = dataset['MSZoning'].map(zoning_mapping)
    dataset['LandContour'] = dataset['LandContour'].map(land_cnt_mapping)
    dataset['LotConfig'] = dataset['LotConfig'].map(lot_cnfg_mapping)
    dataset['LandSlope'] = dataset['LandSlope'].map(land_slope_mapping)
    dataset['Condition1'] = dataset['Condition1'].map(cond1_mapping)
    dataset['Condition1'] = dataset['Condition1'].fillna(0)
    dataset['Condition1'] = dataset['Condition1'].map(int)
    dataset['Condition2'] = dataset['Condition2'].map(cond2_mapping)
    dataset['Condition2'] = dataset['Condition2'].fillna(0)
    dataset['Condition2'] = dataset['Condition2'].map(int)

    dataset['Neighborhood'] = dataset['Neighborhood'].map({
        'MeadowV': 0, 'IDOTRR': 1, 'BrDale': 2,
        'BrkSide': 3, 'Edwards': 4, 'OldTown': 5,
        'Sawyer': 6, 'Blueste': 7,'SWISU': 8,
        'NPkVill': 9, 'NAmes': 10, 'Mitchel': 11,
        'Mitchel': 12, 'SawyerW': 13, 'NWAmes': 14,
        'Gilbert': 15, 'Blmngtn': 16, 'CollgCr': 17,
        'Crawfor': 18, 'ClearCr': 19, 'Somerst': 20,
        'Veenker': 21, 'Timber': 22,
        'StoneBr': 23, 'NridgHt': 24, 'NoRidge': 25
    })

    dataset['RoofStyle'] = dataset['RoofStyle'].map({
        'Shed': 0, 'Mansard': 1, 'Hip': 2,
        'Gambrel': 3, 'Gable': 4, 'Flat': 5,
    })
    dataset['Exterior1st'] = dataset['Exterior1st'].map({
        'CBlock': 1, 'AsphShn': 1, 'ImStucc': 1,
        'Stone': 1, 'BrkComm': 1, 'AsbShng': 1,
        'Stucco': 1, 'WdShing': 1, 'BrkFace': 1,
        'CemntBd': 2, 'Plywood': 3, 'Wd Sdng': 4,
        'MetalSd': 5, 'HdBoard': 6, 'VinylSd': 7
    })
    dataset['Exterior2nd'] = dataset['Exterior2nd'].map({
        'CBlock': 1, 'AsphShn': 1, 'ImStucc': 1,
        'Stone': 1, 'BrkComm': 1, 'AsbShng': 1,
        'Stucco': 1, 'WdShing': 1, 'BrkFace': 1,
        'CemntBd': 2, 'Plywood': 3, 'Wd Sdng': 4,
        'MetalSd': 5, 'HdBoard': 6, 'VinylSd': 7
    })


    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(0)
    dataset['Exterior1st'] = dataset['Exterior1st'].map(int)

    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(0)
    dataset['Exterior2nd'] = dataset['Exterior2nd'].map(int)

    dataset['MasVnrType'] = dataset['MasVnrType'].map({
        'None': 0, 'BrkFace': 1, 'Stone': 2,
        'BrkCmn': 3
    })
    dataset['MasVnrType'] = dataset['MasVnrType'].fillna(0)
    dataset['MasVnrType'] = dataset['MasVnrType'].map(int)

    dataset['BsmtQual'] = dataset['BsmtQual'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })
    dataset['BsmtCond'] = dataset['BsmtCond'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })
    dataset['ExterCond'] = dataset['ExterCond'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })
    dataset['ExterQual'] = dataset['ExterQual'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })

    dataset['BsmtCond'] = dataset['BsmtCond'].fillna(0)
    dataset['BsmtCond'] = dataset['BsmtCond'].map(int)

    dataset['ExterQual'] = dataset['ExterQual'].fillna(0)
    dataset['ExterQual'] = dataset['ExterQual'].map(int)

    dataset['BsmtQual'] = dataset['BsmtQual'].fillna(0)
    dataset['BsmtQual'] = dataset['BsmtQual'].map(int)

    dataset['ExterCond'] = dataset['ExterCond'].fillna(0)
    dataset['ExterCond'] = dataset['ExterCond'].map(int)

    dataset['Foundation'] = dataset['Foundation'].map({
        'Wood': 1, 'Stone': 2, 'Slab': 3,
        'BrkTil': 4, 'CBlock': 5, 'PCinc': 6
    })
    dataset['Foundation'] = dataset['Foundation'].fillna(0)
    dataset['Foundation'] = dataset['Foundation'].map(int)

    dataset['HeatingQC'] = dataset['HeatingQC'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })
    dataset['HeatingQC'] = dataset['HeatingQC'].fillna(0)
    dataset['HeatingQC'] = dataset['HeatingQC'].map(int)

    dataset['CentralAir'] = dataset['CentralAir'].map({
        'N': 0, 'Y': 1
    })

    dataset['Functional'] = dataset['Functional'].map({
        'Typ': 1, 'Sev': 2,'Mod': 3,
        'Min2': 4,'Min1': 5, 'Maj2': 6,
        'Maj1': 7
    })
    dataset['Functional'] = dataset['Functional'].fillna(0)
    dataset['Functional'] = dataset['Functional'].map(int)


    dataset['KitchenQual'] = dataset['KitchenQual'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(rnd.uniform(dataset['KitchenQual'].mean() - dataset['KitchenQual'].std(), dataset['KitchenQual'].mean() + dataset['KitchenQual'].std()))
    dataset['KitchenQual'] = dataset['KitchenQual'].map(int)

    dataset['FireplaceQu'] = dataset['FireplaceQu'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })
    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(0)
    dataset['FireplaceQu'] = dataset['FireplaceQu'].map(int)

    dataset['GarageType'] = dataset['GarageType'].map({
        'Attchd': 1, 'Detchd': 2, 'BuiltIn': 3,
        'Basment': 4, 'CarPort': 5, '2Types': 6,
    })
    dataset['GarageType'] = dataset['GarageType'].fillna(0)
    dataset['GarageType'] = dataset['GarageType'].map(int)

    dataset['GarageQual'] = dataset['GarageQual'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })
    dataset['GarageCond'] = dataset['GarageCond'].map({
        'Po': 1, 'TA': 2, 'Fair': 3,
        'Gd': 4, 'Ex': 5,
    })

    dataset['GarageQual'] = dataset['GarageQual'].fillna(0)
    dataset['GarageQual'] = dataset['GarageQual'].map(int)

    dataset['GarageCond'] = dataset['GarageCond'].fillna(0)
    dataset['GarageCond'] = dataset['GarageCond'].map(int)

    dataset['SaleCondition'] = dataset['SaleCondition'].map({
        'Normal': 1, 'Partial': 2, 'Abnorml': 3,
        'Family': 4, 'Alloca': 5, 'AdjLand': 6
    })

    dataset['SaleType'] = dataset['SaleType'].map({
        'WD': 1, 'New': 2, 'COD': 3,
        'ConLD': 4, 'ConLI': 5, 'ConLw': 6, 'CWD': 7
        , 'Oth': 8, 'Con': 9
    })
    dataset['SaleType'] = dataset['SaleType'].fillna(rnd.uniform(dataset['SaleType'].mean() - dataset['SaleType'].std(), dataset['SaleType'].mean() + dataset['SaleType'].std()))
    dataset['SaleType'] = dataset['SaleType'].map(int)

    # Checking LotFrontage - Numerical data
    # Missing - 259 (or not available option for some houses)
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(0)
    dataset.loc[dataset['LotFrontage'] <= 50, 'LotFrontage'] = 1
    dataset.loc[(dataset['LotFrontage'] > 50) & (dataset['LotFrontage'] <= 100), 'LotFrontage'] = 2
    dataset.loc[(dataset['LotFrontage'] > 100) & (dataset['LotFrontage'] <= 150), 'LotFrontage'] = 3
    dataset.loc[(dataset['LotFrontage'] > 150) & (dataset['LotFrontage'] <= 200), 'LotFrontage'] = 4
    dataset.loc[(dataset['LotFrontage'] > 200) & (dataset['LotFrontage'] <= 250), 'LotFrontage'] = 5
    dataset.loc[dataset['LotFrontage'] > 300, 'LotFrontage'] = 6
    dataset['LotFrontage'] = dataset['LotFrontage'].map(int)

    # Checking LotArea - Numerical data
    # LotArea / SalePrice correlation ?
    # Good correlation between sale price and lot area
    dataset.loc[dataset['LotArea'] <= 40000, 'LotArea'] = 1
    dataset.loc[(dataset['LotArea'] > 40000) & (dataset['LotArea'] <= 80000), 'LotArea'] = 2
    dataset.loc[(dataset['LotArea'] > 80000) & (dataset['LotArea'] <= 120000), 'LotArea'] = 3
    dataset.loc[(dataset['LotArea'] > 120000) & (dataset['LotArea'] <= 160000), 'LotArea'] = 4
    dataset.loc[(dataset['LotArea'] > 160000) & (dataset['LotArea'] <= 200000), 'LotArea'] = 5
    dataset.loc[dataset['LotArea'] > 200000, 'LotArea'] = 6

    dataset.loc[dataset['YearBuilt'] <= 1900, 'YearBuilt'] = 1
    dataset.loc[(dataset['YearBuilt'] > 1900) & (dataset['YearBuilt'] <= 1920), 'YearBuilt'] = 2
    dataset.loc[(dataset['YearBuilt'] > 1920) & (dataset['YearBuilt'] <= 1940), 'YearBuilt'] = 3
    dataset.loc[(dataset['YearBuilt'] > 1940) & (dataset['YearBuilt'] <= 1960), 'YearBuilt'] = 4
    dataset.loc[(dataset['YearBuilt'] > 1960) & (dataset['YearBuilt'] <= 1980), 'YearBuilt'] = 5
    dataset.loc[dataset['YearBuilt'] > 1980, 'YearBuilt'] = 6

    dataset.loc[dataset['YearRemodAdd'] <= 1900, 'YearRemodAdd'] = 1
    dataset.loc[(dataset['YearRemodAdd'] > 1900) & (dataset['YearRemodAdd'] <= 1920), 'YearRemodAdd'] = 2
    dataset.loc[(dataset['YearRemodAdd'] > 1920) & (dataset['YearRemodAdd'] <= 1940), 'YearRemodAdd'] = 3
    dataset.loc[(dataset['YearRemodAdd'] > 1940) & (dataset['YearRemodAdd'] <= 1960), 'YearRemodAdd'] = 4
    dataset.loc[(dataset['YearRemodAdd'] > 1960) & (dataset['YearRemodAdd'] <= 1980), 'YearRemodAdd'] = 5
    dataset.loc[dataset['YearRemodAdd'] > 1980, 'YearRemodAdd'] = 6

    dataset.loc[dataset['MasVnrArea'] <= 320, 'MasVnrArea'] = 1
    dataset.loc[(dataset['MasVnrArea'] > 320) & (dataset['MasVnrArea'] <= 640), 'MasVnrArea'] = 2
    dataset.loc[(dataset['MasVnrArea'] > 640) & (dataset['MasVnrArea'] <= 960), 'MasVnrArea'] = 3
    dataset.loc[(dataset['MasVnrArea'] > 960) & (dataset['MasVnrArea'] <= 1280), 'MasVnrArea'] = 4
    dataset.loc[(dataset['MasVnrArea'] > 1280), 'MasVnrArea'] = 5
    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
    dataset['MasVnrArea'] = dataset['MasVnrArea'].map(int)

    dataset.loc[dataset['TotalBsmtSF'] <= 1000, 'TotalBsmtSF'] = 1
    dataset.loc[(dataset['TotalBsmtSF'] > 1000) & (dataset['TotalBsmtSF'] <= 2000), 'TotalBsmtSF'] = 2
    dataset.loc[(dataset['TotalBsmtSF'] > 2000) & (dataset['TotalBsmtSF'] <= 3000), 'TotalBsmtSF'] = 3
    dataset.loc[(dataset['TotalBsmtSF'] > 3000) & (dataset['TotalBsmtSF'] <= 4000), 'TotalBsmtSF'] = 4
    dataset.loc[(dataset['TotalBsmtSF'] > 4000) & (dataset['TotalBsmtSF'] <= 6500), 'TotalBsmtSF'] = 5
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(0)
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].map(int)

    dataset.loc[dataset['GrLivArea'] <= 334, 'GrLivArea'] = 1
    dataset.loc[(dataset['GrLivArea'] > 334) & (dataset['GrLivArea'] <= 1440), 'GrLivArea'] = 2
    dataset.loc[(dataset['GrLivArea'] > 1440) & (dataset['GrLivArea'] <= 2500), 'GrLivArea'] = 3
    dataset.loc[(dataset['GrLivArea'] > 2500) & (dataset['GrLivArea'] <= 3500), 'GrLivArea'] = 4
    dataset.loc[(dataset['GrLivArea'] > 3500) & (dataset['GrLivArea'] <= 4500), 'GrLivArea'] = 5
    dataset.loc[(dataset['GrLivArea'] > 4500), 'GrLivArea'] = 6

    dataset.loc[dataset['GarageYrBlt'] <= 1900, 'GarageYrBlt'] = 1
    dataset.loc[(dataset['GarageYrBlt'] > 1900) & (dataset['GarageYrBlt'] <= 1920), 'GarageYrBlt'] = 2
    dataset.loc[(dataset['GarageYrBlt'] > 1920) & (dataset['GarageYrBlt'] <= 1940), 'GarageYrBlt'] = 3
    dataset.loc[(dataset['GarageYrBlt'] > 1940) & (dataset['GarageYrBlt'] <= 1960), 'GarageYrBlt'] = 4
    dataset.loc[(dataset['GarageYrBlt'] > 1960) & (dataset['GarageYrBlt'] <= 1980), 'GarageYrBlt'] = 5
    dataset.loc[dataset['GarageYrBlt'] > 1980, 'GarageYrBlt'] = 6
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].map(int)

    dataset.loc[(dataset['GarageArea'] >= 1) & (dataset['GarageArea'] <= 300), 'GarageArea'] = 1
    dataset.loc[(dataset['GarageArea'] > 300) & (dataset['GarageArea'] <= 600), 'GarageArea'] = 2
    dataset.loc[(dataset['GarageArea'] > 600) & (dataset['GarageArea'] <= 900), 'GarageArea'] = 3
    dataset.loc[(dataset['GarageArea'] > 900) & (dataset['GarageArea'] <= 1200), 'GarageArea'] = 4
    dataset.loc[(dataset['GarageArea'] > 1200) & (dataset['GarageArea'] <= 1450), 'GarageArea'] = 5

    dataset.loc[(dataset['WoodDeckSF'] >= 1) & (dataset['WoodDeckSF'] <= 170), 'WoodDeckSF'] = 1
    dataset.loc[(dataset['WoodDeckSF'] > 170) & (dataset['WoodDeckSF'] <= 340), 'WoodDeckSF'] = 2
    dataset.loc[(dataset['WoodDeckSF'] > 340) & (dataset['WoodDeckSF'] <= 520), 'WoodDeckSF'] = 3
    dataset.loc[(dataset['WoodDeckSF'] > 520) & (dataset['WoodDeckSF'] <= 690), 'WoodDeckSF'] = 4
    dataset.loc[(dataset['WoodDeckSF'] > 690) & (dataset['WoodDeckSF'] <= 880), 'WoodDeckSF'] = 5

    dataset.loc[(dataset['OpenPorchSF'] >= 1) & (dataset['OpenPorchSF'] <= 110), 'OpenPorchSF'] = 1
    dataset.loc[(dataset['OpenPorchSF'] > 110) & (dataset['OpenPorchSF'] <= 220), 'OpenPorchSF'] = 2
    dataset.loc[(dataset['OpenPorchSF'] > 220) & (dataset['OpenPorchSF'] <= 320), 'OpenPorchSF'] = 3
    dataset.loc[(dataset['OpenPorchSF'] > 320) & (dataset['OpenPorchSF'] <= 440), 'OpenPorchSF'] = 4
    dataset.loc[(dataset['OpenPorchSF'] > 440) & (dataset['OpenPorchSF'] <= 550), 'OpenPorchSF'] = 5

    dataset.loc[(dataset['EnclosedPorch'] >= 1) & (dataset['EnclosedPorch'] <= 110), 'EnclosedPorch'] = 1
    dataset.loc[(dataset['EnclosedPorch'] > 110) & (dataset['EnclosedPorch'] <= 220), 'EnclosedPorch'] = 2
    dataset.loc[(dataset['EnclosedPorch'] > 220) & (dataset['EnclosedPorch'] <= 330), 'EnclosedPorch'] = 3
    dataset.loc[(dataset['EnclosedPorch'] > 330) & (dataset['EnclosedPorch'] <= 440), 'EnclosedPorch'] = 4
    dataset.loc[(dataset['EnclosedPorch'] > 440) & (dataset['EnclosedPorch'] <= 555), 'EnclosedPorch'] = 5

    dataset.loc[dataset['YrSold'] == 2006, 'YrSold'] = 1
    dataset.loc[dataset['YrSold'] == 2007, 'YrSold'] = 2
    dataset.loc[dataset['YrSold'] == 2008, 'YrSold'] = 3
    dataset.loc[dataset['YrSold'] == 2009, 'YrSold'] = 4
    dataset.loc[dataset['YrSold'] == 2010, 'YrSold'] = 5

    dataset.loc[dataset['MSSubClass'] == 20, 'MSSubClass'] = 1
    dataset.loc[dataset['MSSubClass'] == 30, 'MSSubClass'] = 2
    dataset.loc[dataset['MSSubClass'] == 40, 'MSSubClass'] = 3
    dataset.loc[dataset['MSSubClass'] == 45, 'MSSubClass'] = 4
    dataset.loc[dataset['MSSubClass'] == 50, 'MSSubClass'] = 5
    dataset.loc[dataset['MSSubClass'] == 60, 'MSSubClass'] = 6
    dataset.loc[dataset['MSSubClass'] == 70, 'MSSubClass'] = 7
    dataset.loc[dataset['MSSubClass'] == 75, 'MSSubClass'] = 8
    dataset.loc[dataset['MSSubClass'] == 80, 'MSSubClass'] = 9
    dataset.loc[dataset['MSSubClass'] == 85, 'MSSubClass'] = 10
    dataset.loc[dataset['MSSubClass'] == 90, 'MSSubClass'] = 11
    dataset.loc[dataset['MSSubClass'] == 120, 'MSSubClass'] = 12
    dataset.loc[dataset['MSSubClass'] == 150, 'MSSubClass'] = 13
    dataset.loc[dataset['MSSubClass'] == 160, 'MSSubClass'] = 14
    dataset.loc[dataset['MSSubClass'] == 180, 'MSSubClass'] = 15
    dataset.loc[dataset['MSSubClass'] == 190, 'MSSubClass'] = 16

    dataset['MSZoning'] = dataset['MSZoning'].fillna(0)
    dataset['MSZoning'] = dataset['MSZoning'].map(int)

    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].map(int)
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0)
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].map(int)
    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
    dataset['GarageCars'] = dataset['GarageCars'].map(int)
    dataset['GarageArea'] = dataset['GarageArea'].fillna(0)
    dataset['GarageArea'] = dataset['GarageArea'].map(int)

    dataset['ExtoriorComb'] = dataset['Exterior2nd'] + dataset['Exterior1st']
    dataset['Condition12'] =  dataset['Condition1'] + dataset['Condition2']


    dataset.drop(['Id', 'Street', 'Alley', 'Utilities',
                  'BldgType', 'HouseStyle', 'RoofMatl',
                  'PoolQC', 'Fence', 'MiscFeature',
                  'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                  'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
                  'Heating', '1stFlrSF', '2ndFlrSF',
                  'LowQualFinSF', 'GarageFinish', 'PavedDrive',
                  '3SsnPorch', 'ScreenPorch', 'PoolArea', 'Exterior1st', 'Exterior2nd','Condition1','Condition2','MasVnrType', 'MasVnrArea'], axis=1, inplace=True)


total_miss = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending=False)
miss_data = pd.concat([total_miss, percent], axis=1, keys=['Total', 'Percent'])


print(miss_data.head(20))


corr_mat = train_df.corr()
col = corr_mat.columns.values


# print(train_df.info())
# print(test_df.info())
# print(col)


cat_cols = train_df.select_dtypes(include='object',).columns.values
num_cols = train_df.select_dtypes(include=['int64' , 'float64']).columns.values

# print(cat_cols)
# print(num_cols)

# Cehck corrs > 0.5 and < - 0.5 pairs


def corr_table(table):
    for row in table:
        for col in table:
            if 0.5 < table.at[row,col] < 1:
                print(row,col, corr_mat.at[row,col])
            if -1 < corr_mat.at[row,col] < -0.5:
                print(row,col, corr_mat.at[row,col])

#orr_table(corr_mat)

# corr_table(corr_mat)

both_data = [train_df, test_df]

print(train_df.shape, test_df.shape)


#
# print(train_df.info())
# print(test_df.info())
# print(train_df.head())



# Get X parameters for and Y output separated
X_train = train_df.drop("SalePrice", axis=1)
Y_train = train_df["SalePrice"]
X_test = test_df.copy()

# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
#
#
# print(coeff_df.sort_values(by='Correlation', ascending=False))
# print(acc_log)
# print(mean_squared_log_error(Y_train[1:], Y_pred))
#
# grboostregress = GradientBoostingRegressor()
# grboostregress.fit(X_train, Y_train)
# Y_pred = grboostregress.predict(X_test)
# print(mean_squared_log_error(Y_train[1:], Y_pred))
#
#
# knnr = KNeighborsRegressor()
# knnr.fit(X_train, Y_train)
# Y_pred = knnr.predict(X_test)
# print(mean_squared_log_error(Y_train[1:], Y_pred))
#
#
#

