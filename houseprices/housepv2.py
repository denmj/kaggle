# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

#pre precessing
from sklearn import preprocessing as prep

# ml models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor


# Evaluation
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score







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


def normalize(x):
    f_mean = x.mean()
    f_sigma = x.std()
    x_norm = (x - f_mean) / f_sigma
    return x_norm


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
pass_id_test = test_df["Id"]

# Quick check for empty vals in  cols
# Drop cols with more than 50% empty cells

train_df.drop(["Id", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], axis=1, inplace=True)
test_df.drop(["Id", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], axis=1, inplace=True)
# Get median of LotFrontage and replace NaN
median_lotFr_tr = train_df['LotFrontage'].median()
train_df['LotFrontage'].replace(np.nan, median_lotFr_tr, inplace=True)
test_df['LotFrontage'].replace(np.nan, median_lotFr_tr, inplace=True)

# Get avg and replace missing vals
avg_bs_sf1 = np.round(train_df['BsmtFinSF1'].mean())
avg_bs_sf = np.round(train_df['BsmtUnfSF'].mean())
avg_tbs_sf = np.round(train_df['TotalBsmtSF'].mean())
avg_garage_area = np.round(train_df['GarageArea'].mean())

test_df['BsmtFinSF1'].replace(np.nan, avg_bs_sf1, inplace=True)
test_df['BsmtUnfSF'].replace(np.nan, avg_bs_sf, inplace=True)
test_df['TotalBsmtSF'].replace(np.nan, avg_tbs_sf, inplace=True)
test_df['GarageArea'].replace(np.nan, avg_garage_area, inplace=True)


# Replace missing values by Frequency
train_df['Electrical'].replace(np.nan, 'SBrkr', inplace=True)
test_df['MSZoning'].replace(np.nan, 'RL', inplace=True)
test_df['Utilities'].replace(np.nan, 'AllPub', inplace=True)
test_df['Functional'].replace(np.nan, 'Typ', inplace=True)
test_df['Exterior1st'].replace(np.nan, 'VinylSd', inplace=True)
test_df['Exterior2nd'].replace(np.nan, 'VinylSd', inplace=True)
test_df['KitchenQual'].replace(np.nan, 'TA', inplace=True)
test_df['GarageCars'].replace(np.nan, 2, inplace=True)
test_df['SaleType'].replace(np.nan, 'WD', inplace=True)




# Replace missing values in Garage Year build by 0
# replace by 0 and NaN MasVnr features
for col in ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF2', 'GarageYrBlt', 'MasVnrArea']:
    test_df[col].replace(np.nan, 0, inplace=True)
    train_df[col].replace(np.nan, 0, inplace=True)

# Replace missing values by none in categorical vars for garage
# Basement, same missing values for each basement related
# feature, we will assume that basement is not available for those houses
# Replace by "none" all NaN values
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType', 'BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']:
    train_df[col].replace(np.nan, 'None', inplace=True)
    test_df[col].replace(np.nan, 'None', inplace=True)

# Check for missing vals
# tr =  missing_zero_values_table(train_df)
# t = missing_zero_values_table(test_df)

# Correlation matrix
# corr_mat(train_df)

# details = stats(train_df, 'SalePrice')
# print(details.sort_values(by='corr SalePrice', ascending=False))


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

# Visualize missing values
# sns.heatmap(train_df.isnull(), cbar=False)
# plt.show()
# sns.heatmap(test_df.isnull(), cbar=False)
# plt.show()

# Perform binning

# Encode categorical data into num.ordinal
cat_cols = train_df.select_dtypes(include='object').columns.values
enc = prep.OrdinalEncoder()
enc.fit(train_df[cat_cols])
train_df[cat_cols] = enc.transform(train_df[cat_cols])
test_df[cat_cols] = enc.transform(test_df[cat_cols])


# Scaling
# train_df['SalePrice'] = np.log1p(train_df['SalePrice'])


y_train = np.asarray(train_df["SalePrice"])
X_train = np.asarray(train_df.drop("SalePrice", axis=1))
X_test = np.asarray(test_df)

X_train = prep.RobustScaler().fit(X_train).transform(X_train)
X_test = prep.RobustScaler().fit(X_test).transform(X_test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

# Target variable
# plt.subplots(figsize=(12, 9))
# sns.distplot(train_df['SalePrice'], fit=scipy.stats.norm)
# plt.show()


# visualization of some data (Features with corr > 0.5)
# fig = plt.figure(figsize=(45, 25))
# sns.set(font_scale=2)

# (Corr= 0.790982) Box plot overallqual/salePrice
# fig1 = fig.add_subplot(331)
# sns.boxplot(x='OverallQual', y='SalePrice',  data=train_df[['SalePrice', 'OverallQual']])

# # Next one
# fig2 = fig.add_subplot(332)
# sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'GrLivArea', 'OverallQual']])
#
# fig3 = fig.add_subplot(333)
# sns.scatterplot(x='TotalBsmtSF', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'TotalBsmtSF', 'OverallQual']])
#
# fig4 = fig.add_subplot(334)
# sns.boxplot(x='GarageCars', y='SalePrice',  data=train_df[['SalePrice', 'GarageCars', 'OverallQual']])
#
# fig5 = fig.add_subplot(335)
# sns.scatterplot(x='GarageArea', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', 'GarageArea', 'OverallQual']])
#
# fig6 = fig.add_subplot(336)
# sns.scatterplot(x='1stFlrSF', y='SalePrice', hue='OverallQual', data=train_df[['SalePrice', '1stFlrSF', 'OverallQual']])
#
# fig7 = fig.add_subplot(337)
# sns.boxplot(x='FullBath', y='SalePrice', data=train_df[['SalePrice', 'FullBath']])
#
# fig8 = fig.add_subplot(338)
# sns.boxplot(x='TotRmsAbvGrd', y='SalePrice', data=train_df[['SalePrice', 'TotRmsAbvGrd']])
#
# fig9 = fig.add_subplot(339)
# sns.scatterplot(x='YearBuilt', y='SalePrice', data=train_df[['SalePrice', 'YearBuilt']])
#
#
# plt.show()
#
# top_corr_features_col = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'GarageArea', '1stFlrSF']
# sns.set(style='ticks')
# sns.pairplot(train_df[top_corr_features_col], height=3, kind='reg')
# plt.show()

# Models

# Trying different models

xbreg = XGBRegressor()
xb_model = xbreg.fit(X_train, y_train)
acc_xb_train = round(xb_model.score(X_train, y_train) * 100, 2)
acc_xb_val = round(xb_model.score(X_val, y_val) * 100, 2)
pred_xb_tr = xb_model.predict(X_train)
pred_xb_val = xb_model.predict(X_val)

rmlse_train_xb = np.sqrt(mean_squared_log_error(y_train, pred_xb_tr))
rmlse_val_xb = np.sqrt(mean_squared_log_error(y_val, pred_xb_val))


lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)
acc_lr_train = round(lr_model.score(X_train, y_train) * 100, 2)
acc_lr_val = round(lr_model.score(X_val, y_val) * 100, 2)
pred_lr_tr = lr_model.predict(X_train)
pred_lr_val = lr_model.predict(X_val)
pred_lr_test = lr_model.predict(X_test)
print(pred_lr_tr)
print(pred_lr_val)
print(pred_lr_test)

rmlse_train_lr = np.sqrt(mean_squared_log_error(y_train, pred_lr_tr))
rmlse_val_lr = np.sqrt(mean_squared_log_error(y_val, pred_lr_val))



print(rmlse_train_xb, rmlse_val_xb)
print("XB for train set: ", acc_xb_train)
print("XB for val set: ", acc_xb_val)



print(rmlse_train_lr, rmlse_val_lr)
print("LR for train set: ", acc_lr_train)
print("LR for val set: ", acc_lr_val)

# Best model
test_y = lr_model.predict(X_test)
test_Y = xb_model.predict(X_test)
print(test_y)
print(test_Y)

submission = pd.DataFrame({
        "Id": pass_id_test,
        "SalePrice": test_Y
    })

submission.to_csv('submission_3_0.csv', index=False)

