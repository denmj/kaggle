# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt

# ml models
from sklearn.linear_model import LogisticRegression

# Panda options
pd.set_option('display.max_columns', 20)


# checks missing values in dataset
def check_miss_values(data):
    miss_vals = data.isnull()
    for column in miss_vals.columns.values.tolist():
        print(column)
        print(miss_vals[column].value_counts())


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Check data set for missing values
# Drop 'Cabin' col
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)

# Drop Ticket column - doesn't tell any useful information
train_df.drop("Ticket", axis=1, inplace=True)
test_df.drop("Ticket", axis=1, inplace=True)


# print(train_df.head())
# print(test_df.head())

# Dealing with missing values in dataset
# Same in both data sets
avg_age = train_df["Age"].mean().round()

train_df["Age"].replace(np.nan, avg_age, inplace=True)
test_df["Age"].replace(np.nan, avg_age, inplace=True)


check_miss_values(train_df)
check_miss_values(test_df)

# Max frequent value in a column and replace NaN with it
print(train_df["Embarked"].value_counts())
train_df["Embarked"].replace(np.nan, "S", inplace=True)
