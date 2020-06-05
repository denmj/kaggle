import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1) for conv
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

print(x_train.shape)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)



# Build the model.
model = Sequential([
  Dense(500, activation='relu', input_shape=(784,)),
  Dropout(0.25),
  Dense(500, activation='relu'),
  Dropout(0.25),
  Dense(500, activation='relu'),
  Dropout(0.25),
  Dense(500, activation='relu'),
  Dropout(0.25),
  Dense(500, activation='relu'),
  Dropout(0.25),
  Dense(10, activation='softmax'),
])

model.summary()
# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model_5h = model.fit(
  x_train,
  y_train,
  epochs=10,
  batch_size=128,
  validation_split=0.2
)

