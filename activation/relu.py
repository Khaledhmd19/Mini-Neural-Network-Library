import numpy as np
from .activation import Activation

class ReLU(Activation):
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.input > 0)
