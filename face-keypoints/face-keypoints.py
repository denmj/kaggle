import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def display_pcs(pics):
    for i in range(10):
        img = pics[i].reshape(96, 96)
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.show()


def split_image_data(image_data):
    imag = []
    for i in range(0, 7049):
        img = image_data['Image'][i].split(' ')
        img = ['0' if x == '' else x for x in img]
        imag.append(img)

    return np.asarray(imag, dtype='float')


train_df = pd.read_csv('data/training.csv')
test_df = pd.read_csv('data/test.csv')
cols = train_df.columns.values
ids_df = pd.read_csv('data/IdLookupTable.csv')

images_array = split_image_data(train_df)
img = images_array.reshape(-1, 96, 96, 1)
display_pcs(img[2:12])




