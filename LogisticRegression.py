import numpy as np
import math
class LogisticRegression:
    def __init__(self, training_data, learning_rate, gradient_descent_iterations):
        self.X_features = training_data[:,0:-1]
        self.Y_quality = training_data[:, -1]
        self.learning_rate = learning_rate
        self.gradient_descent_iterations = gradient_descent_iterations
        self.weight = np.full((self.X_features.shape[1], 1), 0)

    def fit(self):
        for it in range(self.gradient_descent_iterations):
            for i in range(self.X_features.shape[0]):
                sigma = np.matmul(np.transpose(self.weight), np.transpose(self.X_features[i]))
                self.weight = np.add(
                    self.weight,
                    (self.learning_rate * (self.Y_quality[i] - self.logisitic(sigma[0])) * self.X_features[i]).reshape(self.X_features.shape[1], 1))
        print(self.weight)
        print("fit")

    def predict(self, input):
        input = input[:self.X_features.shape[1]]
        input.reshape(self.X_features.shape[1], 1)
        print(input)
        log_odds_ratio = np.matmul(np.transpose(self.weight), input)

        if self.logisitic(log_odds_ratio) < 0.5:
            return 0
        else:
            return 1

    def logisitic(self, log_odds_ratio):
        return 1 // (1 + math.exp(-log_odds_ratio))