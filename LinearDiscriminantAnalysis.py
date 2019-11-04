import numpy as np
import math
class LinearDiscriminantAnalysis:
    def __init__(self, X_features, Y_quality):
        self.X_features = X_features
        self.Y_outcomes = Y_quality

        # preprocess
        self.estimate_parameter()

    def fit(self):
        self.w0 = np.log(self.p[1]/self.p[0]) + 1/2*np.dot(np.dot(np.array(self.u0), np.linalg.inv(self.matrix)), np.array(self.u0).reshape(-1,1)) - 1/2*np.dot(np.dot(np.array(self.u1), np.linalg.inv(self.matrix)), np.array(self.u1).reshape(-1,1))
        self.w = np.dot(np.linalg.inv(self.matrix), np.array([x1 - x2 for (x1, x2) in zip(self.u1, self.u0)]).reshape(-1,1));

    def predict(self, input):
        res = []
        for x in input:
            val = self.w0 + np.dot(x, self.w)
            if val > 0:
                res.append(1)
            else:
                res.append(0)
        return res

    def estimate_parameter(self):
        self.N0 = 0
        self.N1 = 0
        for i in range(self.Y_outcomes.shape[0]):
            if self.Y_outcomes[i] == 0:
                self.N0 = self.N0 + 1
            else:
                self.N1 = self.N1 + 1
        # estimate P(y)
        self.p = np.array([self.N0 / (self.N0 + self.N1), self.N1 / (self.N0 + self.N1)])
        # estimate u
        self.u0 = [0] * (np.size(self.X_features, 1))
        self.u1 = [0] * (np.size(self.X_features, 1))
        class0 = []
        class1 = []

        for index, data in enumerate(self.X_features):
            if self.Y_outcomes[index] == 0:
                self.u0 = [sum(x) for x in zip(self.u0, data)]
                class0.append(data)
            if self.Y_outcomes[index] == 1:
                self.u1 = [sum(x) for x in zip(self.u1, data)]
                class1.append(data)

        self.u0[:] = [x / self.N0 for x in self.u0]
        self.u1[:] = [x / self.N1 for x in self.u1]
        class0[:] = [[x1 - x2 for (x1, x2) in zip(x, self.u0)] for x in class0]
        class1[:] = [[x1 - x2 for (x1, x2) in zip(x, self.u1)] for x in class1]

        # estimate corvariane
        matrix0 = np.zeros((len(class0[0]), len(class0[0])))
        matrix1 = np.zeros((len(class1[0]), len(class1[0])))

        for x in class0:
            matrix0 = np.add(matrix0, (np.array(x).reshape((-1, 1)) @ np.array(x).reshape(1, -1)))
        for x in class1:
            matrix1 = np.add(matrix1, (np.array(x).reshape((-1, 1)) @ np.array(x).reshape(1, -1)))

        self.matrix = np.divide(np.add(matrix0, matrix1), np.size(self.Y_outcomes) - 2)

        # print(np.linalg.inv(self.matrix))

