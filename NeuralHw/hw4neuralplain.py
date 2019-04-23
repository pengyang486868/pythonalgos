import numpy as np
import time
import math

# read in data and weights
# since label data is in fact 1-10, we will change them into 0 to 9
X = np.genfromtxt('ps5_data.csv', delimiter=',')
y = np.genfromtxt('ps5_data-labels.csv', delimiter=',')
y = y - 1
W1 = np.genfromtxt('ps5_theta1.csv', delimiter=',')
W2 = np.genfromtxt('ps5_theta2.csv', delimiter=',')
print(X.shape, y.shape, W1.shape, W2.shape)


## 4.2 softmax and sigmoid activation
def softmax(yhat):
    # 4.2

    return result


def sigmoid(z):
    return


def toCategorical(y):
    # convert to one hot

    return y_one_hot


class NeuralNetwork:
    def __init__(self):
        self.W1 = None  # W1 should be 401 * 25
        self.W2 = None  # W2 should be 26 * 10

    def setWeight(self, W1, W2):

    def neuronActivation(self, Aprev, W, with_sigmoid=True):
        # 4.1
        # this will take activation of previous layer and the weight of previous layer to this layer
        # to produce a single activation value

        return a

    def final_layer(self, activations):
        # 4.3 activation of the final layer

        return probabilities

    def forward(self, X):
        # 4.4
        # non-vectorized. Given 1 example, return a prediction

        return probabilities

    def classify(self, X):
        # 4.5, should use 4.4 forward function
        # will classify an image and return a number

        return int(predicted_label)

    def get_error_rate(self, X, y):
        # 4.6 will give error rate on given X, y
        # you can use python's time
        start_time = time.time()

        return error_rate, end_time - start_time

    def giveCost(self, X, y):
        # 4.7
        # gives the value of the cross entropy cost function

        return J

nn = NeuralNetwork()
nn.setWeight(W1.transpose(), W2.transpose())
error_rate, time_used = nn.get_error_rate(X, y)
cost = nn.giveCost(X, y)
print(error_rate, cost, time_used)