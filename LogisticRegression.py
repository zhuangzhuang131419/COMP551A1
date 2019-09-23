import numpy as np
import math
class LogisticRegression:
    def __init__(self, X_features, Y_quality):
        # insert x0 = 1

        self.X_features = np.insert(X_features, 0, values=1, axis=1)
        # print(X_features)
        # print(self.X_features)
        # self.X_features = X_features
        self.Y_quality = Y_quality
        self.weight = np.full((self.X_features.shape[1], 1), 0)

        # normalize the x_feature
        # for i in range(self.X_features.shape[1]):
        #     self.X_features[:, i] = self.normalize(self.X_features[:, i])

        # print(self.X_features)

    def normalize(self, x):
        if max(x) == min(x):
            return x
        return (x) / (max(x) - min(x))

    def fit(self, learning_rate, gradient_descent_iterations):
        for it in range(gradient_descent_iterations):
            weight_old = self.weight
            for i in range(self.X_features.shape[0]):
                sigma = np.dot(np.transpose(weight_old), np.transpose(self.X_features[i]))
                self.weight = self.weight + learning_rate * (self.Y_quality[i] - self.logisitic(sigma)) * (self.X_features[i]).reshape(self.weight.shape[0], 1)
        # print(it, self.weight)

    def predict(self, input):
        target_evaluation = np.full((input.shape[0], 1), -1)
        input = np.insert(input, 0, values=1, axis=1)
        # print(self.weight)
        for i in range(input.shape[0]):
            log_odds_ratio = np.dot(np.transpose(self.weight), input[i])
            target_evaluation[i] = 0 if self.logisitic(log_odds_ratio) < 0.5 else 1
        return target_evaluation

    def logisitic(self, log_odds_ratio):
        if log_odds_ratio < 0:
            return 1 - 1 / (1 + math.exp(log_odds_ratio))
        else:
            return 1 / (1 + math.exp(-log_odds_ratio))