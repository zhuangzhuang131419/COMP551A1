#!/usr/bin/python
import numpy as np
import pandas as pd
# Visualisation
import matplotlib.pyplot as plt

### Task1
# 1.Load the datasets into numpy pbjects
red_wine_data = np.genfromtxt('wine/winequality-red.csv', delimiter =';', skip_header = True)
# red_wine_data = pd.read_csv('wine/winequality-red.csv', delimiter=';')
breast_cancer_data = np.genfromtxt('breast/breast-cancer-wisconsin.data', delimiter = ',', skip_header = True)
# breast_cancer_data = pd.read_csv('breast/breast-cancer-wisconsin.data', delimiter=',', header = None)


# 2.Convert the wine dataset to a binary task
# for index, row in red_wine_data.iterrows():
#     if red_wine_data.loc[index, 'quality'] < 6:
#         red_wine_data.loc[index, 'quality'] = 0
#     else:
#         red_wine_data.loc[index, 'quality'] = 1
rows, cols = red_wine_data.shape
for i in range(rows):
    if red_wine_data[i][-1] < 6:
        red_wine_data[i][-1] = 0
    else:
        red_wine_data[i][-1] = 1

# 3.Clean the data
red_wine_data = red_wine_data[~np.isnan(red_wine_data).any(axis=1)]
breast_cancer_data = breast_cancer_data[~np.isnan(breast_cancer_data).any(axis=1)]
# 4.Compute some statistics on the data.
print(red_wine_data)

### Task2



