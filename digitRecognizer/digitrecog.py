from time import time

# data analysis and wrangling
import pandas as pd
import numpy as np

# Standard scientific Python imports
import matplotlib.pyplot as plt

# compression
from sklearn.decomposition import PCA

# model
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# prepossessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# metrics
from sklearn import metrics

# data
from sklearn import datasets

pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 150)


def display_digits(digits, labels):
    for i in range(25):
        img = digits[i].reshape(28, 28)
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % labels[i])
    plt.show()


def principal_component_plot(pca):
    t0 = time()
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.title('Pulsar Dataset Explained Variance')
    plt.show()
    print("Number of Principal components for PCA done in %0.3fs" % (time() - t0))
    cum_var_per = pca.explained_variance_ratio_.cumsum()
    number_of_comp = len(cum_var_per[cum_var_per <= 0.90])
    print(" Within 90% - {}".format(number_of_comp))


t0 = time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print("Loading data done in %0.3fs" % (time() - t0))

t0 = time()
y_train = np.asarray(train_df["label"])
X_train = np.asarray(train_df.drop("label", axis=1)).astype('float32')
X_test = np.asarray(test_df).astype('float32')
print("Converting to numpy array in %0.3fs" % (time() - t0))

# check digits
# display_digits(X_train, y_train)

# lets try with simple split
t0 = time()
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
print("Splitting data done in %0.3fs" % (time() - t0))

# Using portion of data (for quicker computations)
i = 1000
print(train_df.shape)

t0 = time()
sc = StandardScaler().fit(X_tr)
X_std_tr = sc.transform(X_tr)
X_std_val = sc.transform(X_val)
print("Normalizing data done in %0.3fs" % (time() - t0))

# Picking best number of components for PCA
pca_func = PCA().fit(X_std_tr)
principal_component_plot(pca_func)

t0 = time()
pca = PCA(n_components=225).fit(X_std_tr)
train_pca = pca.transform(X_std_tr[:i])
val_pca = pca.transform(X_std_val)
print("PCA done in %0.3fs" % (time() - t0))
print("Shape before PCA for Train: ", X_std_tr.shape)
print("Shape after PCA for Train: ", train_pca.shape)
print("Shape before PCA for Test: ", X_std_val.shape)
print("Shape after PCA for Test: ", val_pca.shape)

# Picking  best C and gamma for SVM
svm_params = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

# clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
#                    svm_params, cv=5, iid=False)


# number of neighbors
knn_params = {'n_neighbors': [2, 3, 5, 7, 10, 15, 20],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']
              }

# knnclf = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, n_jobs=-1)

# KNN
t0 = time()
print("Fitting KNN...")
knnclf = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
knnclf.fit(train_pca, y_tr[:i])
print("Fitting KNN done in %0.3fs" % (time() - t0))

# SVM
t0 = time()
print("Fitting SVM...")
clf = svm.SVC(C=1000, gamma=0.001, random_state=None)
clf.fit(train_pca, y_tr[:i])
print("Fitting SVM done in %0.3fs" % (time() - t0))
t0 = time()
score_svm = clf.score(val_pca, y_val)
score_knn = knnclf.score(val_pca, y_val)

# print("Best estimator found by grid search:")
# print(knnclf.best_estimator_)
# print("Best score found by grid search:")
# print(knnclf.best_score_)
# print("Best params found by grid search:")
# print(knnclf.best_params_)

print("SVM accuracy : ", score_svm)
print("KNN accuracy : ", score_knn)

print("Accuracy calculated in %0.3fs" % (time() - t0))

print('Done training')

expected_X_val = y_val
predicted_X_val = clf.predict(val_pca)

# Test data and submission

print("Full size training data and test data")
y_t = np.asarray(train_df["label"])
X_t = np.asarray(train_df.drop("label", axis=1)).astype('float32')
X_tt = np.asarray(test_df).astype('float32')

t0 = time()
sc = StandardScaler().fit(X_t)
X_std_t = sc.transform(X_t)
X_std_tt = sc.transform(X_tt)
print("Normalizing data for full data (train and test)  done in %0.3fs" % (time() - t0))

print("Performing PCA...")
t0 = time()
pca_ = PCA(n_components=225).fit(X_std_t)
t_pca = pca_.transform(X_std_t)
tt_pca = pca_.transform(X_std_tt)
print("PCA done in %0.3fs" % (time() - t0))

print("Fitting SVM on Full data...")
clf_ = svm.SVC(C=1000, gamma=0.001, random_state=None)
clf_.fit(t_pca, y_t)
print("Fitting SVM done in %0.3fs" % (time() - t0))

print("Predicting test data...")
result = clf_.predict(tt_pca)
print(result)


submission_data = pd.DataFrame( {
    'ImageId': test_df.index.values+1,
    'Label': result
})

submission_data.index = submission_data['ImageId'].values

print(submission_data.head())

submission_data.to_csv('/Users/denismamedov/PycharmProjects/kaggle/digitRecognizer/data/result.csv', index = False)
