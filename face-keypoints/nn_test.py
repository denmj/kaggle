import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_reshaped = x_train.reshape(x_train.shape[0], 28, 28, -1).astype('float32')


# Data augmentation
data_aug = ImageDataGenerator()

data_aug.fit(x_train_reshaped)
print(x_train_reshaped.shape)

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

#
# model_h5_plain = Sequential([
#   Dense(500, activation='relu', input_shape=(784,)),
#   Dense(500, activation='relu'),
#   Dense(500, activation='relu'),
#   Dense(500, activation='relu'),
#   Dense(500, activation='relu'),
#   Dense(10, activation='softmax'),
#
# ])
#
# model_h5_dropout = Sequential([
#   Dense(500, activation='relu', input_shape=(784,)),
#   Dropout(0.25),
#   Dense(500, activation='relu'),
#   Dropout(0.25),
#   Dense(500, activation='relu'),
#   Dropout(0.25),
#   Dense(500, activation='relu'),
#   Dropout(0.25),
#   Dense(500, activation='relu'),
#   Dropout(0.25),
#   Dense(10, activation='softmax'),
# ])
#
# model_h5_l1 = Sequential([
#   Dense(500, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l1(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),
#   Dense(10, activation='softmax'),
#
# ])
#
# model_h5_l2 = Sequential([
#   Dense(500, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#   Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#   Dense(10, activation='softmax'),
#
# ])
#
# model_h5_plain.summary()
# model_h5_dropout.summary()
# model_h5_l1.summary()


# # Compile the model.
# model_h5_dropout.compile(
#   optimizer='adam',
#   loss='categorical_crossentropy',
#   metrics=['accuracy'],
# )

# Train the model.
# model_5h_droupout = model_h5_dropout.fit(
#   x_train,
#   y_train,
#   epochs=10,
#   batch_size=128,
#   validation_split=0.2
# )

