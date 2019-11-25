from __future__ import absolute_import, division, print_function, unicode_literals

from time import time
# plot and graph
import matplotlib.pyplot as plt

# data analysis and wrangling
import pandas as pd

# from keras.models import Sequential

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from sklearn.model_selection import train_test_split


t0 = time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print("Loading data done in %0.3fs" % (time() - t0))

img_rows, img_cols = 28, 28
num_classes = 10

# Train data
X_test = test_df.values.astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)


def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


raw_data = pd.read_csv('data/train.csv')
x, y = data_prep(raw_data)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y,
          batch_size=64,
          epochs=3,
          validation_split=0.2)

# predictions = model.predict_classes(X_test, verbose=0)
#
# submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
#                          "Label": predictions})
# submissions.to_csv("cnn_tf.csv", index=False, header=True)