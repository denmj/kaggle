import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pre processing
from sklearn.impute import SimpleImputer

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Splits Image array and keypoint coordinates
def split_image_data(image_data):
    imag = []
    for i in range(0, 7049):
        img = image_data['Image'][i].split(' ')
        img = ['0' if x == '' else x for x in img]
        imag.append(img)

    return np.asarray(imag, dtype='float')


# Displays array as an image
def display_pcs(df):
    images_array = split_image_data(df)
    img_reshape = images_array.reshape(-1, 96, 96, 1)
    for i in range(10):
        img = img_reshape[i].reshape(96, 96)
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.show()


def missing_values(df):
    miss_vals = df.isnull().sum()
    miss_percent = 100 * miss_vals / len(df)
    miss_values_table = pd.concat([miss_vals, miss_percent], axis=1)
    miss_values_table = miss_values_table.rename(
        columns={0: 'Missing Values', 1: '% of Missing Values'}
    )

    print(miss_values_table)


train_df = pd.read_csv('data/training.csv')
test_df = pd.read_csv('data/test.csv')
cols = train_df.columns.values
ids_df = pd.read_csv('data/IdLookupTable.csv')

# Display images
# display_pcs(train_df)

# print(train_df.describe())
# missing_values(train_df)

# we could use fillna with ffill method,
# that will fill in last valid value observed
# train_df.fillna(method='ffill', inplace=True)

# Or replace NaNs with mean()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
train_image = split_image_data(train_df)

y_train = imputer.fit_transform(train_df[cols[:-1]].to_numpy())

print(train_image.shape)
print(y_train.shape)

X_train = train_image.reshape(-1, 96, 96, 1)
print(X_train.shape)


# Define model
model = Sequential([Flatten(input_shape=(96, 96, 1)),
                         Dense(128, activation="relu"),
                         Dropout(0.1),
                         Dense(64, activation="relu"),
                         Dense(30)
                         ])

# model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
#
# model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
#
# model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
#
# model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# # model.add(BatchNormalization())
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
#
# model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
#
# model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
#
# model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
#
#
# model.add(Flatten())
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(30))
model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])


model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2)
