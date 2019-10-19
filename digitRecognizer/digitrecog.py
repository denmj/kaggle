# data analysis and wrangling
import pandas as pd
import numpy as np
from PIL import Image

pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 150)


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

y_train = np.asarray(train_df["label"])
X_train = np.asarray(train_df.drop("label", axis=1)).astype('float32')
X_test = np.asarray(test_df).astype('float32')

print(X_train[0].shape)
print(X_train[0].reshape(28, 28))

im = Image.fromarray(X_train[5].reshape(28, 28))
im.show()

