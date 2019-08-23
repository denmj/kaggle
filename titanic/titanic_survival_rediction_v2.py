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
# Drop Ticket and Id column - doesn't tell any useful information
train_df.drop(["Ticket", "PassengerId", "Cabin"], axis=1, inplace=True)
test_df.drop(["Ticket", "PassengerId", "Cabin"], axis=1, inplace=True)

# Dealing with missing values in dataset
# Same in both data sets
avg_age = train_df["Age"].mean().round()

train_df["Age"].replace(np.nan, avg_age, inplace=True)
test_df["Age"].replace(np.nan, avg_age, inplace=True)


# Max frequent value in a column and replace NaN with it
# print(train_df["Embarked"].value_counts())
train_df["Embarked"].replace(np.nan, "S", inplace=True)

# 1 value is missing in Fare column and we will fill it by mean value.
avg_fare = test_df["Fare"].mean()
test_df["Fare"].replace(np.nan, avg_fare, inplace=True)

pclass_sex = train_df[['Survived', 'Sex', 'Pclass']]
pclass_sex = pclass_sex.groupby(['Sex', 'Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(pclass_sex)

# Combine Siblings and Parent columns into one
train_df["FamSize"] = train_df["SibSp"] + train_df["Parch"]
test_df["FamSize"] = test_df["SibSp"] + test_df["Parch"]
train_df.drop(["SibSp", "Parch"], axis=1, inplace=True)
test_df.drop(["SibSp", "Parch"], axis=1, inplace=True)


fam_group = train_df[["FamSize", "Survived"]]
fam_group = fam_group.groupby(["FamSize"], as_index=False).mean().sort_values(by="Survived", ascending=False)
# print(fam_group)


class_age_surv = sns.FacetGrid(train_df, col="Survived", row="Pclass")
class_age_surv.map(plt.hist, "Age")
plt.show()

# Transform categorical features into ordinal
embark_enc = prep.OrdinalEncoder()
embark_enc.fit(train_df[["Embarked"]])
train_df["Embarked"] = embark_enc.transform(train_df[["Embarked"]])
test_df["Embarked"] = embark_enc.transform(test_df[["Embarked"]])
sex_enc = prep.OrdinalEncoder()
sex_enc.fit(train_df[["Sex"]])
train_df["Sex"] = sex_enc.transform(train_df[["Sex"]])
test_df["Sex"] = sex_enc.transform(test_df[["Sex"]])


# Binning Age and Fare data
bin_age = prep.KBinsDiscretizer(n_bins=7, encode='ordinal').fit(train_df[["Age"]])
train_df["Age"] = bin_age.transform(train_df[["Age"]])
test_df["Age"] = bin_age.transform(test_df[["Age"]])

bin_fare = prep.KBinsDiscretizer(n_bins=5, encode='ordinal').fit(train_df[["Fare"]])
train_df["Fare"] = bin_fare.transform(train_df[["Fare"]])
test_df["Fare"] = bin_fare.transform(test_df[["Fare"]])

print(train_df.head())

# Use titles extracted from Name column and break into categories
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# check_miss_values(train_df)
# check_miss_values(test_df)
# print(train_df.head(5))
#
print(train_df.corr())

print(train_df.dtypes)


print(train_df["Title"].value_counts())