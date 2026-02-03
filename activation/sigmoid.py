import numpy as np
from .activation import Activation

class Sigmoid(Activation):
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, grad_output):
        s = self.output
        return grad_output * s * (1 - s)
