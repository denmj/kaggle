# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt

#pre precessing
from sklearn import preprocessing as prep


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


# Some useful funcs
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
t = missing_zero_values_table(train_df)
print(t)

corr_mat(train_df)

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

# visualization of some data

fig = plt.figure(figsize=(25, 20))
sns.set(font_scale=2)

# (Corr= 0.790982) Box plot overallqual/salePrice
fig1 = fig.add_subplot(221)
sns.boxplot(x='OverallQual', y='SalePrice',  data=train_df[['SalePrice', 'OverallQual']])
# Next one
fig2 = fig.add_subplot(222)
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'GrLivArea', 'OverallQual']])

fig3 = fig.add_subplot(223)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'TotalBsmtSF', 'OverallQual']])

fig4 = fig.add_subplot(224)
sns.boxplot(x='GarageCars', y='SalePrice',  data=train_df[['SalePrice', 'GarageCars', 'OverallQual']])


plt.show()
