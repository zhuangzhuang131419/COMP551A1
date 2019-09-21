#!/usr/bin/python
import numpy as np
import LogisticRegression as LR
# Visualisation
import matplotlib.pyplot as plt

### Task1
# 1.Load the datasets into numpy pbjects
red_wine_data = np.genfromtxt('wine/winequality-red.csv', delimiter =';', skip_header = True)
# red_wine_data = pd.read_csv('wine/winequality-red.csv', delimiter=';')
breast_cancer_data = np.genfromtxt('breast/breast-cancer-wisconsin.data', delimiter = ',', skip_header = True)
# breast_cancer_data = pd.read_csv('breast/breast-cancer-wisconsin.data', delimiter=',', header = None)
# print(red_wine_data)
# print(breast_cancer_data)

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
# print(red_wine_data[:,0:-1])

### Task2
def evaluate_acc(X_feature, Y_true_label, Y_target_label):
    print("evaluate_accuracy")
    total = 0
    for i in range(Y_true_label.shape[0]):
        if Y_true_label[i] == Y_target_label[i]:
            total = total + 1
    print(total)
    return total / Y_true_label.shape[0]



# implement with k-fold cross validation
learning_rate = 0.001
gradient_descent_iterations = 100
# initialize the target array
Y_quality_target = np.full((rows, 1), 0)
# for i in range(rows):
#     # let ith row be test data
#     X_features = red_wine_data[:, 0:-1]
#     Y_quality = red_wine_data[:, -1]
#     X_features_training = np.delete(X_features, i, axis = 0)
#     Y_quality_training = np.delete(Y_quality, i, axis = 0)
#
#     logistic_regression = LR.LogisticRegression(X_features, Y_quality)
#     logistic_regression.fit(learning_rate, gradient_descent_iterations)
#
#     Y_quality_target[i] = logistic_regression.predict(X_features[i])
#
# print(evaluate_acc(red_wine_data, Y_quality, Y_quality_target))


logistic_regression = LR.LogisticRegression(red_wine_data[:, 0:-1], red_wine_data[:, -1])
logistic_regression.fit(learning_rate, gradient_descent_iterations)
for i in range(rows):
    Y_quality_target[i] = logistic_regression.predict(red_wine_data[i])

print(Y_quality_target)

print(red_wine_data[:, -1].reshape(rows, 1))
print(evaluate_acc(red_wine_data, red_wine_data[:, -1].reshape(rows, 1), Y_quality_target))
# print(evaluate_acc(red_wine_data, red_wine_data[:, -1].reshape(rows, 1), Y_quality_target))
