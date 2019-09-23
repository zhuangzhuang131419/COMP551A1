#!/usr/bin/python
import numpy as np
import LogisticRegression as LR
import LinearDiscriminantAnalysis as LDA
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

for i in range(red_wine_data.shape[0]):
    if red_wine_data[i][-1] < 6:
        red_wine_data[i][-1] = 0
    else:
        red_wine_data[i][-1] = 1

for i in range(breast_cancer_data.shape[0]):
    if breast_cancer_data[i][-1] == 2:
        breast_cancer_data[i][-1] = 0
    else:
        breast_cancer_data[i][-1] = 1

# discard the first column(numerical order)
breast_cancer_data = breast_cancer_data[:, 1:]


# 3.Clean the data
red_wine_data = red_wine_data[~np.isnan(red_wine_data).any(axis=1)]
breast_cancer_data = breast_cancer_data[~np.isnan(breast_cancer_data).any(axis=1)]
# 4.Compute some statistics on the data.


### Task2
def evaluate_acc(X_feature, Y_true_label, Y_target_label):
    total = 0
    for i in range(Y_true_label.shape[0]):
        if Y_true_label[i] == Y_target_label[i]:
            total = total + 1
    return total / Y_true_label.shape[0]

### Task 3
# implement with k-fold cross validation
learning_rate = 0.01
gradient_descent_iterations = 100
def k_fold(data, k):
    np.random.shuffle(data)
    data_subsets = np.array_split(data, k, axis = 0)
    avg_acc = 0
    for i in range(k):
        training_data = np.concatenate(data_subsets[:i] + data_subsets[i + 1:], axis = 0)
        validation_data = data_subsets[i]
        logistic_regression_k_fold = LR.LogisticRegression(training_data[:, 0:-1], training_data[:, -1])
        logistic_regression_k_fold.fit(learning_rate, gradient_descent_iterations)
        y_perdict = logistic_regression_k_fold.predict(validation_data[:, 0:-1])
        avg_acc = avg_acc + evaluate_acc(data_subsets, validation_data[:, -1].reshape(validation_data.shape[0], 1), y_perdict)
    print("Logestic average accurrcy: ", avg_acc / k)
    avg_acc = 0
    for i in range(k):
        training_data = np.concatenate(data_subsets[:i] + data_subsets[i + 1:], axis=0)
        validation_data = data_subsets[i]
        linear_discriminant_analysis_k_fold = LDA.LinearDiscriminantAnalysis(training_data[:, 0:-1], training_data[:, -1])
        linear_discriminant_analysis_k_fold.fit()
        y_perdict = linear_discriminant_analysis_k_fold.predict(validation_data[:, 0:-1])

        avg_acc = avg_acc + evaluate_acc(data_subsets, validation_data[:, -1].reshape(validation_data.shape[0], 1), y_perdict)
    print("LDA average accurrcy:", avg_acc / k)
    return

k_fold(red_wine_data, 5)
k_fold(breast_cancer_data, 5)