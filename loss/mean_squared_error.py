import numpy as np
from .loss import Loss

class MeanSquaredError(Loss):
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target) ** 2)

    def backward(self, grad_output=None):
        N = self.prediction.shape[0]
        grad = 2 * (self.prediction - self.target) / N
        return grad
