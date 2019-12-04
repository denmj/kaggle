import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pre processing
from sklearn.impute import SimpleImputer

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
train_keypoints = imputer.fit_transform(train_df[cols[:-1]].to_numpy())

print(train_image.shape)
print(train_keypoints.shape)

