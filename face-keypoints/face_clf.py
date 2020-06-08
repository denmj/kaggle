import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv



# ml models
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

# pre precessing
from sklearn import preprocessing as prep

# Panda settings
pd.set_option('display.max_columns', 20)
sns.set(style="ticks")


def data_insight(data, y_label=None):
    types = data.dtypes
    counts = data.apply(lambda x: x.count())
    nulls = data.apply(lambda x: x.isna().sum())
    skewness = data.skew()
    kurtosis = data.kurt()
    unique_count = data.apply(lambda x: x.unique().shape[0])
    unique_list = data.apply(lambda x: [x.unique()])

    if y_label is None:
        cols = ['types', 'counts', 'nulls', 'unique counts', 'unique list', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, nulls, unique_count, unique_list, skewness, kurtosis], axis=1)
        str.columns = cols
    print(str)
    print('\n Descriptive stats: ')
    print(data.describe())


def pic_and_points(X, Y):
    for i in range(4):
        rand_pic = np.random.randint(low=0, high=X.shape[0])
        img = X[rand_pic]
        plt.subplot(2, 2, i+1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        for c in range(0, 30, 2):
            plt.plot(int(Y.iloc[rand_pic][c]), int(Y.iloc[rand_pic][c+1]), 'r+')

# Load data


train_df = pd.read_csv('C:/Users/denis/Desktop/ML/kaggle_data/face/training.csv')
test_df = pd.read_csv('C:/Users/denis/Desktop/ML/kaggle_data/face/test.csv')
idLookUpTable = pd.read_csv('C:/Users/denis/Desktop/ML/kaggle_data/face/IdLookupTable.csv')

columns = train_df.columns
# Filling missing values with next valid entry
train_df.fillna(method='ffill', inplace=True)


train_Y = train_df[columns[:-1]]
train_X = train_df[columns[-1]]

# Reshape
test_df['Image'] = test_df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))

test_X = np.asarray([test_df['Image']], dtype=np.uint8).reshape(test_df.shape[0], 96, 96, 1)
train_X = train_X.apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
train_X = np.asarray([train_X], dtype=np.uint8).reshape(train_X.shape[0], 96, 96, 1)
train_Y = train_Y.to_numpy()


print(train_X.shape, train_Y.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Convolution2D(32, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Convolution2D(64, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Convolution2D(64, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Convolution2D(96, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Convolution2D(96, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Convolution2D(128, (3, 3), padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())
#
model.add(tf.keras.layers.Convolution2D(128, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
#
model.add(tf.keras.layers.Convolution2D(256, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Convolution2D(256, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Convolution2D(512, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Convolution2D(512, (3, 3), padding='same', use_bias=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(30))
model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

hist = model.fit(train_X, train_Y, epochs=50, batch_size=256, validation_split=0.2)

model.save('C:/Users/denis/Desktop/ML/K/kaggle/face-keypoints')


