import random

import numpy as np


class Multi_Perceptron:

    def __init__(self):
        self.weights = []
        self.bias = []

    def _winit(self):
        return random.random()

    def _sigmoid(self, H):
        return 1.0/(1+np.exp(-H))

    def _loss(self, y, yhat):
        return -np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    def _gradient(self):
        pass

    def add_layer(self, _in, _out):
        self.weights.append(np.full((_out, _in), 1))
        self.bias.append(np.full((_out, 1), 1))

    def fit(self, x, y, iter=10):

        # iteration = 10
        for i in range(iter):
            hypo = None

            for widx in range(len(self.weights)):
                # hypothesis 계산
                hypo = np.dot(self.weights[widx], x) + self.bias[widx]
                hypo = self._sigmoid(hypo)

            # loss cal
            loss = self._loss(y, hypo)
            print(hypo)
            print(loss)

            # gradient descent
            self._gradient(hypo)




    def predict(self, x):
        pass



data = np.array([[0,0,1,1],[0,1,0,1]])

print(data.shape)
print(data.shape[1])
label = np.array([[0,1,1,0]])

# 인스턴스 선언
mp = Multi_Perceptron()

# 층
mp.add_layer(2, 2)
mp.add_layer(2, 1)
# print(np.dot(mp.weights[0], data))
mp.fit(data, label)