from time import time

# data analysis and wrangling
import pandas as pd
import numpy as np

# Standard scientific Python imports
import matplotlib.pyplot as plt

t0 = time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print("Loading data done in %0.3fs" % (time() - t0))

print(train_df.shape)
print(test_df.shape)
