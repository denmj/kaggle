# data analysis and wrangling
import pandas as pd
import numpy as np
# visualization package
import seaborn as sns
import matplotlib.pyplot as plt

#pre precessing
from sklearn import preprocessing as prep

# ml models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm


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
bin_age = prep.KBinsDiscretizer(n_bins=5, encode='ordinal').fit(train_df[["Age"]])
train_df["Age"] = bin_age.transform(train_df[["Age"]])
test_df["Age"] = bin_age.transform(test_df[["Age"]])

bin_fare = prep.KBinsDiscretizer(n_bins=5, encode='ordinal').fit(train_df[["Fare"]])
train_df["Fare"] = bin_fare.transform(train_df[["Fare"]])
test_df["Fare"] = bin_fare.transform(test_df[["Fare"]])

# Use titles extracted from Name column and break into categories
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_df['Title'] = train_df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

test_df['Title'] = test_df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')

title_enc = prep.OrdinalEncoder()
title_enc.fit(train_df[["Title"]])
train_df["Title"] = title_enc.transform(train_df[["Title"]])
test_df["Title"] = title_enc.transform(test_df[["Title"]])

train_df.drop(["Name"], axis=1, inplace=True)
test_df.drop(["Name"], axis=1, inplace=True)


print(train_df.corr())

# Model
# X_train = train_df.drop("Survived", axis=1)
# Y_train = train_df["Survived"]
# X_test = test_df.drop("PassengerId", axis=1).copy()

X_train = np.asarray(train_df.drop("Survived", axis=1))
y_train = np.asarray(train_df["Survived"])
X_out_of_sample = np.asarray(test_df)

print(X_train[0:5])
print(y_train[0:5])
print(X_out_of_sample[0:5])

X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.3, random_state=4)


X_train = prep.StandardScaler().fit(X_train).transform(X_train)
print(X_train[0:5])


# Trying different models
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)
print(lr)
y_test_hat_lr = lr.predict(X_out_of_sample)
acc_lr = round(lr.score(X_train, y_train) * 100, 2)


svm = svm.SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_test_hat_svm = svm.predict(X_out_of_sample)
acc_svm = round(svm.score(X_train, y_train) * 100, 2)


print("LogReg: ", acc_lr)
print("SVM: ", acc_svm)
