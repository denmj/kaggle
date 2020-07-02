import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Pre processing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# CONSTANTS
TRAIN_IMAGE_PATH = "C:/Users/denis/Desktop/ML/kaggle_data/dogs-vs-cats/train"
TEST_IMAGE_PATH = "C:/Users/denis/Desktop/ML/kaggle_data/dogs-vs-cats/test1"
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 16
NUM_CLASSES = 2
CHANNELS = (3,)
EPOCHS = 50


def images_to_df(f_path):
    f_names = os.listdir(f_path)
    categories = []
    for f_name in f_names:
        klass = f_name.split('.')[0]
        if klass == 'dog':
            categories.append('dog')
        else:
            categories.append('cat')

    df = pd.DataFrame({
        'filename': f_names,
        'categories': categories
    })
    return df


df_dataset = images_to_df(TRAIN_IMAGE_PATH)


train, val = train_test_split(df_dataset, test_size=0.1)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(train, TRAIN_IMAGE_PATH, x_col='filename',
                                                    y_col='categories', class_mode='binary',
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=BATCH_SIZE)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(val, TRAIN_IMAGE_PATH, x_col='filename',
                                                    y_col='categories', class_mode='binary',
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=BATCH_SIZE)

#
# example_df = train.sample(n=1)
#
# print(example_df)
#
# ex_gen = train_datagen.flow_from_dataframe(
#     example_df,
#     TRAIN_IMAGE_PATH,
#     x_col='filename',
#     y_col='categories',
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE)
#
#
# plt.figure(figsize=(12, 12))
# for i in range(0, 4):
#     plt.subplot(1, 4, i+1)
#     for X_batch, Y_batch in ex_gen:
#         image = X_batch[0]
#         plt.imshow(image)
#         break
# plt.tight_layout()
# plt.show()


# Xeption mini
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


# model = make_model(input_shape=IMAGE_SIZE+CHANNELS, num_classes=2)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
# ]
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss="binary_crossentropy",
#     metrics=["accuracy"],
# )
# model.fit_generator(
#     train_generator, epochs=EPOCHS, callbacks=callbacks, validation_data=val_generator,
# )
#
# model.save("Xeption_catdog")


loaded_model = keras.models.load_model('C:/Users/denis/Desktop/ML/K/kaggle/cats_dogs_kaggle/Xeption_catdog')

Y_val = val['categories']
loss, accuracy = loaded_model.evaluate_generator(val_generator)

print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))

test_filenames = os.listdir(TEST_IMAGE_PATH)
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    TEST_IMAGE_PATH,
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=BATCH_SIZE,
    target_size=IMAGE_SIZE,
    shuffle=False
)

predict = loaded_model.predict(test_generator, steps=np.ceil(nb_samples/BATCH_SIZE))
threshold = 0.5
test_df['category'] = np.where(predict > threshold, 1, 0)


sample_test = test_df.sample(n=9).reset_index()
sample_test.head()
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("C:/Users/denis/Desktop/ML/kaggle_data/dogs-vs-cats/test1/"+filename, target_size=(256, 256))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()
