from time import time

# data analysis and wrangling
import pandas as pd
import numpy as np
from PIL import Image

# Standard scientific Python imports
import matplotlib.pyplot as plt

# compression
from sklearn.decomposition import PCA

# model
from sklearn.model_selection import GridSearchCV
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

t0 = time()
sc = StandardScaler().fit(X_tr)
X_std_tr = sc.transform(X_tr)
X_std_val = sc.transform(X_val)
print("Normalizing data done in %0.3fs" % (time() - t0))


t0 = time()
pca = PCA(n_components=300).fit(X_std_tr)
train_pca = pca.transform(X_std_tr)
val_pca = pca.transform(X_std_val)
print("PCA done in %0.3fs" % (time() - t0))

print("Shape before PCA for Train: ",X_std_tr.shape)
print("Shape after PCA for Train: ",train_pca.shape)
print("Shape before PCA for Test: ",X_std_val.shape)
print("Shape after PCA for Test: ",val_pca.shape)

t0 = time()
clf = svm.SVC(C=10, gamma=0.001, random_state=42)
clf.fit(train_pca, y_tr)
print("Fitting done in %0.3fs" % (time() - t0))

t0 = time()
score=clf.score(val_pca, y_val)
print("Accuracy : ", score)
print("Accuracy calculated in %0.3fs" % (time() - t0))

# learn digits on training data
# clf = svm.SVC(gamma=0.001)
# clf.fit(X_tr, y_tr)
print('Done training')

# expected_X_val = y_tr
# predicted_X_val = clf.predict(X_val)





#
# n_components = 150
# print("Extracting the top %d eigenfaces from %d faces"
#       % (n_components, X_tr.shape[0]))
# t0 = time()
# pca = PCA(n_components=n_components, svd_solver='randomized',
#           whiten=True).fit(X_tr)
# print("done in %0.3fs" % (time() - t0))
#
# # eigenfaces = pca.components_.reshape((n_components, h, w))
#
# print("Projecting the input data on the eigenfaces orthonormal basis")
# t0 = time()
# X_train_pca = pca.transform(X_tr)
# X_test_pca = pca.transform(X_val)
# print("done in %0.3fs" % (time() - t0))
#
#
# print("Fitting the classifier to the training set")
# t0 = time()
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
#                    param_grid, cv=5, iid=False)
#
# clf = clf.fit(X_train_pca, y_tr)
#
# print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)
