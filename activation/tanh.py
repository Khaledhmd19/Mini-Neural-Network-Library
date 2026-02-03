import numpy as np
from .activation import Activation

class Tanh(Activation):
    def forward(self, input_data):
        self.input = input_data
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)
