# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt

#pre precessing
from sklearn import preprocessing as prep


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


print(train_df.dtypes)

print(train_df.shape)
print(test_df.shape)

print("All cols names :", train_df.columns)
print("cols with nan: ", train_df.columns[train_df.isnull().any()])
print("number of NaN:", train_df["MSSubClass"].isnull().sum())
# print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
print(train_df.describe())

# Check for missing vals in cols
# cols_with_missing_vals = train_df.columns[train_df.isnull().any()]
# print(cols_with_missing_vals)
