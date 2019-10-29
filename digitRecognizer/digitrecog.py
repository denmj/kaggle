# data analysis and wrangling
import pandas as pd
import numpy as np
from PIL import Image

# Standard scientific Python imports
import matplotlib.pyplot as plt

# model
from sklearn import svm

# metrics
from sklearn import metrics

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


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(train_df.shape,test_df.shape)

y_train = np.asarray(train_df["label"])
print(y_train.shape)

X_train = np.asarray(train_df.drop("label", axis=1)).astype('float32')
X_test = np.asarray(test_df).astype('float32')

display_digits(X_train, y_train)


# print(X_train[0].shape)
# print(X_train[0].reshape(28, 28))
#
# im = Image.fromarray(X_train[5].reshape(28, 28))
# im.show()

