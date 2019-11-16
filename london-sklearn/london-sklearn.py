from time import time

# data analysis and wrangling
import pandas as pd
import numpy as np

# Standard scientific Python imports
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

t0 = time()
train_df = pd.read_csv('data/train.csv')
train_l = pd.read_csv('data/trainLabels.csv')
test_df = pd.read_csv('data/test.csv')
print("Loading data done in %0.3fs" % (time() - t0))

# 1.Scale data
# 2.binaries and categorize


corr_matrix = train_df.corr()

t0 = time()
pipe_line = Pipeline([('std_scaler', StandardScaler())])
x_train_prepared = pipe_line.fit_transform(train_df)
x_train, x_val, y_train, y_val = train_test_split(x_train_prepared, train_l.to_numpy(), test_size=0.3, random_state=42)
print("Preparing data done in %0.3fs" % (time() - t0))

t0 = time()
model_names = ['KNN', 'SVC', 'SGDClassifier']
classifiers = [
    KNeighborsClassifier(),
    SVC(),
    SGDClassifier()]

clf_scores = []
for name, clf in zip(model_names, classifiers):
    clf.fit(x_train, y_train.ravel())
    acc_score_train = round(clf.score(x_train, y_train) * 100, 2)
    acc_score_val = round(clf.score(x_val, y_val) * 100, 2)
    clf_scores.append((name, acc_score_train, acc_score_val))
print("Training and eval done in %0.3fs" % (time() - t0))

for name, acc_t, acc_v in clf_scores:
    print("{} classifier score(r2): [Test data - {}, Validation data - {}] \n".format(name, acc_t, acc_v))
