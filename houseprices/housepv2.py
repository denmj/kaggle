# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

#pre precessing
from sklearn import preprocessing as prep

# Some useful funcs
def corr_mat(df):
    corr = df.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap="YlGnBu",
        square=True,
        linewidths=.5
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()


def stats(df, pred=None):
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum() / obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt()
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis=1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis=1,
                        sort=False)
        corr_col = 'corr ' + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col]

    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\n Data types:\n', str.types.value_counts())
    print('___________________________')
    return str


def missing_zero_values_table(df):
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
        columns={0: 'Zero Values', 1: 'Missing Values', 2: '% of Total Values'})
    mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
    mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"
                                                                                                   "There are " + str(
        mz_table.shape[0]) +
          " columns that have missing values.")
    #         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
    return mz_table


def check_miss_values(df):
    miss_vals = df.isnull()
    for column in miss_vals.columns.values.tolist():
        print(column)
        print(miss_vals[column].value_counts())


pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 150)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
df_id = train_df["Id"]


# Quick check for empty vals in  cols
# Drop cols with more than 50% empty cells

train_df.drop(["Id", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], axis=1, inplace=True)
test_df.drop(["Id", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], axis=1, inplace=True)
# Get median of LotFrontage and replace NaN
median_lotFr = train_df['LotFrontage'].median()
train_df['LotFrontage'].replace(np.nan, median_lotFr, inplace=True)
# Replace missing values in Garage Year build by 0
# replace by 0 and NaN MasVnr features
train_df['GarageYrBlt'].replace(np.nan, 0, inplace=True)
train_df['MasVnrArea'].replace(np.nan, 0, inplace=True)
train_df['Electrical'].replace(np.nan, 'SBrkr', inplace=True)

# Replace missing values by none in categorical vars for garage
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType']:
    train_df[col].replace(np.nan, 'None', inplace=True)

# Basement, same missing values for each basement related
# feature, we will assume that basement is not available for those houses
# Replace by "none" all NaN values
for col in ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']:
    train_df[col].replace(np.nan, 'None', inplace=True)

# Check for missing vals
# t = missing_zero_values_table(train_df)
# print(t)

# Correlation matrix
# corr_mat(train_df)

details = stats(train_df, 'SalePrice')
print(details.sort_values(by='corr SalePrice', ascending=False))


# print("All cols names :", train_df.columns)
# print("cols with nan: ", train_df.columns[train_df.isnull().any()])
# print("number of NaN:", train_df["MSSubClass"].isnull().sum())
# print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
# print(train_df.describe())
# print(train_df.corr())

# Check for missing vals in cols
# cols_with_missing_vals = train_df.columns[train_df.isnull().any()]
# print(cols_with_missing_vals)


# Deal with outliers
train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)


# Perform binning

# Encode categorical data into num.ordinal

# Target val normalization ?

# Feature Engineering ? Create new features if needed

# Target variable
plt.subplots(figsize=(12, 9))
sns.distplot(train_df['SalePrice'], fit=scipy.stats.norm)
plt.show()


# visualization of some data (Features with corr > 0.5)
fig = plt.figure(figsize=(45, 25))
sns.set(font_scale=2)

# (Corr= 0.790982) Box plot overallqual/salePrice
fig1 = fig.add_subplot(331)
sns.boxplot(x='OverallQual', y='SalePrice',  data=train_df[['SalePrice', 'OverallQual']])
# Next one
fig2 = fig.add_subplot(332)
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'GrLivArea', 'OverallQual']])

fig3 = fig.add_subplot(333)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'TotalBsmtSF', 'OverallQual']])

fig4 = fig.add_subplot(334)
sns.boxplot(x='GarageCars', y='SalePrice',  data=train_df[['SalePrice', 'GarageCars', 'OverallQual']])

fig5 = fig.add_subplot(335)
sns.scatterplot(x='GarageArea', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'GarageArea', 'OverallQual']])

fig6 = fig.add_subplot(336)
sns.scatterplot(x='1stFlrSF', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', '1stFlrSF', 'OverallQual']])

fig7 = fig.add_subplot(337)
sns.boxplot(x='FullBath', y='SalePrice', data=train_df[['SalePrice', 'FullBath']])

fig8 = fig.add_subplot(338)
sns.boxplot(x='TotRmsAbvGrd', y='SalePrice', data=train_df[['SalePrice', 'TotRmsAbvGrd']])

fig9 = fig.add_subplot(339)
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train_df[['SalePrice', 'YearBuilt']])


plt.show()

# top_corr_features_col = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'GarageArea', '1stFlrSF']
# sns.set(style='ticks')
# sns.pairplot(train_df[top_corr_features_col], height=3, kind='reg')
# plt.show()

# print(train_df['LotFrontage'].median(), train_df['LotFrontage'].mean())

