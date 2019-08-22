# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt

#pre precessing
from sklearn import preprocessing as prep

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

# Drop Ticket and Id column - doesn't tell any useful information
train_df.drop("Ticket", axis=1, inplace=True)
test_df.drop("Ticket", axis=1, inplace=True)
train_df.drop("PassengerId", axis=1, inplace=True)
test_df.drop("PassengerId", axis=1, inplace=True)

# print(train_df.head())
# print(test_df.head())

# Dealing with missing values in dataset
# Same in both data sets
avg_age = train_df["Age"].mean().round()

train_df["Age"].replace(np.nan, avg_age, inplace=True)
test_df["Age"].replace(np.nan, avg_age, inplace=True)


# Max frequent value in a column and replace NaN with it
print(train_df["Embarked"].value_counts())
train_df["Embarked"].replace(np.nan, "S", inplace=True)

# 1 value is missing in Fare column and we will fill it by mean value.
avg_fare = test_df["Fare"].mean()
test_df["Fare"].replace(np.nan, avg_fare, inplace=True)

check_miss_values(train_df)
check_miss_values(test_df)
print(train_df.head(5))


#
pclass_sex = train_df[['Survived', 'Sex', 'Pclass']]
pclass_sex = pclass_sex.groupby(['Sex', 'Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pclass_sex)

# Combine Siblings and Parent columns into one
train_df["FamSize"] = train_df["SibSp"] + train_df["Parch"]
test_df["FamSize"] = test_df["SibSp"] + test_df["Parch"]
train_df.drop("SibSp", axis=1, inplace=True)
test_df.drop("SibSp", axis=1, inplace=True)
train_df.drop("Parch", axis=1, inplace=True)
test_df.drop("Parch", axis=1, inplace=True)


fam_group = train_df[["FamSize", "Survived"]]
fam_group = fam_group.groupby(["FamSize"], as_index=False).mean().sort_values(by="Survived", ascending=False)
print(fam_group)


class_age_surv = sns.FacetGrid(train_df, col="Survived", row="Pclass")
class_age_surv.map(plt.hist, "Age")
plt.show()


print(train_df.corr())

print(train_df.dtypes)


# print(train_df.dtypes)
