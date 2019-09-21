import numpy as np
import math
class LogisticRegression:
    def __init__(self, training_data, learning_rate, gradient_descent_iterations):
        self.X_features = training_data[:,0:-1]
        self.Y_quality = training_data[:, -1]
        self.learning_rate = learning_rate
        self.gradient_descent_iterations = gradient_descent_iterations
        self.weight = np.full((self.X_features.shape[1], 1), 1)

    def fit(self):
        for it in range(self.gradient_descent_iterations):
            for i in range(self.X_features.shape[0]):
                sigma = np.matmul(np.transpose(self.weight), np.transpose(self.X_features[i]))
                self.weight = np.add(
                    self.weight,
                    self.learning_rate * self.logisitic(sigma[0]) * np.transpose(self.X_features[i]).reshape(self.X_features.shape[1], 1))
            print(self.weight)

        # print(self.X_features[0])
        # print(self.Y_quality[0])
        print("fit")

    def predict(self, input):
        np.matmul(self.weight, input)
        print("predict")

    def logisitic(self, log_odds_ratio):
        return 1 // (1 + math.exp(-log_odds_ratio))