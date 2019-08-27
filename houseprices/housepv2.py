# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt

#pre precessing
from sklearn import preprocessing as prep


# Some useful funcs
def check_miss_values(data):
    miss_vals = data.isnull()
    for column in miss_vals.columns.values.tolist():
        print(column)
        print(miss_vals[column].value_counts())


pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 150)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


print(train_df.shape)
print(test_df.shape)

# Check for missing vals in cols
cols_with_missing_vals = train_df.columns[train_df.isnull().any()]
print(cols_with_missing_vals)