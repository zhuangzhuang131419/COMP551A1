import numpy as np
from sklearn import datasets, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

red_wine_data = np.genfromtxt('wine/winequality-red.csv', delimiter =';', skip_header = True)
breast_cancer_data = np.genfromtxt('breast/breast-cancer-wisconsin.data', delimiter = ',', skip_header = True)

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

breast_cancer_data = breast_cancer_data[:, 1:]
breast_cancer_data = breast_cancer_data[~np.isnan(breast_cancer_data).any(axis=1)]
# print(breast_cancer_data)

def k_fold(data, k):
    np.random.shuffle(data)
    data_subsets = np.array_split(data, k, axis = 0)
    for i in range(k):
        training_data = np.concatenate(data_subsets[:i] + data_subsets[i + 1:], axis = 0)
        validation_data = data_subsets[i]

        # LDA
        clf = LinearDiscriminantAnalysis()
        # print("trainingX", training_data[:, 0:-1])
        # print("trainingY", training_data[:, -1])
        clf.fit(training_data[:, 0:-1], training_data[:, -1])
        diabetes_y_pred = clf.predict(validation_data[:, 0:-1])



        # # Regression
        # regr = linear_model.LinearRegression()
        # regr.fit(training_data[:, 0:-1], training_data[:, -1])
        # diabetes_y_pred = regr.predict(validation_data[:, 0:-1])

        # for i in diabetes_y_pred.shape[1]:
        #     if diabetes_y_pred[i] <
        print(evaluate_acc(validation_data[:, -1].reshape(validation_data.shape[0], 1), diabetes_y_pred))
    return

def evaluate_acc(Y_true_label, Y_target_label):
    print("evaluate_accuracy")
    # print("Y_true_label", Y_true_label)
    # print("Y_target_label", Y_target_label)
    total = 0
    for i in range(Y_true_label.shape[0]):
        # print(Y_true_label[i][0])
        # print(Y_target_label[i][0])
        Y_target_label[i] = 0 if Y_target_label[i] < 0.5 else 1
        if Y_true_label[i] == Y_target_label[i]:
            total = total + 1
    print("total", total)
    return total / Y_true_label.shape[0]

k_fold(red_wine_data, 5)
# k_fold(breast_cancer_data, 5)