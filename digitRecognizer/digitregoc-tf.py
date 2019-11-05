from __future__ import absolute_import, division, print_function, unicode_literals
from time import time

# data analysis and wrangling
import pandas as pd
import numpy as np

# Install TensorFlow
import tensorflow as tf

t0 = time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print("Loading data done in %0.3fs" % (time() - t0))

X_train_k = (train_df.iloc[:,1:].values).astype('float32') # all pixel values
y_train_k = train_df.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test_k = test_df.values.astype('float32')

X_train_k = X_train_k.reshape(-1, 28, 28, 1)
X_test_k = X_test_k.reshape(-1, 28, 28, 1)

# tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_k, y_train_k, epochs=25)

# model.evaluate(x_test,  y_test, verbose=2)