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
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

t0 = time()
train_df = pd.read_csv('data/train.csv', header=None)
train_l = pd.read_csv('data/trainLabels.csv', header=None)
test_df = pd.read_csv('data/test.csv', header=None)
print("Loading data done in %0.3fs" % (time() - t0))

corr_matrix = train_df.corr()
t0 = time()
pipe_line = Pipeline([('std_scaler', StandardScaler())])
x_train_prepared = pipe_line.fit_transform(train_df)
x_train, x_val, y_train, y_val = train_test_split(x_train_prepared, train_l.to_numpy(), test_size=0.3, random_state=42)
print("Preparing data done in %0.3fs" % (time() - t0))

t0 = time()
model_names = ['KNN', 'SVC', 'SGDClassifier',
               'GaussianClf', 'Decision Tree Clf',
               'Random Forest Clf']
classifiers = [
    KNeighborsClassifier(),
    SVC(),
    SGDClassifier(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier()]

clf_scores = []
for name, clf in zip(model_names, classifiers):
    clf.fit(x_train, y_train.ravel())
    acc_score_train = round(clf.score(x_train, y_train) * 100, 2)
    acc_score_val = round(clf.score(x_val, y_val) * 100, 2)
    clf_scores.append((name, acc_score_train, acc_score_val))
print("Training and eval done in %0.3fs" % (time() - t0))

for name, acc_t, acc_v in clf_scores:
    print("{} classifier score(r2): [Test data - {}, Validation data - {}] \n".format(name, acc_t, acc_v))
#
param_grid_rnd_frst = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
               {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

param_grid_svc = [{'C': [0.001, 0.01, 0.1, 1, 10],
               'gamma': [0.001, 0.01, 0.1, 1, 10]}]

grid_search_rnd_frst = GridSearchCV(RandomForestClassifier(), param_grid_rnd_frst, cv=5)
grid_search_rnd_frst.fit(x_train_prepared, train_l.to_numpy().ravel())

grid_search_svc = GridSearchCV(SVC(), param_grid_svc, cv=5)
grid_search_svc.fit(x_train_prepared, train_l.to_numpy().ravel())


for model in [grid_search_svc, grid_search_rnd_frst]:
    score = model.best_score_
    print("Final classifier score(r2): {} \n".format(score))


rand_forest_model = grid_search_rnd_frst.best_estimator_
svc_model = grid_search_svc.best_estimator_

x_test_prepared = pipe_line.fit_transform(test_df)
submission = pd.DataFrame(svc_model.predict(x_test_prepared))
submission.columns = ['Solution']
submission['Id'] = np.arange(1, submission.shape[0]+1)
submission = submission[['Id', 'Solution']]
submission.to_csv("data/submission_svc_with_normalization.csv", index=False)
