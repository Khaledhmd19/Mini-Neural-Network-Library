import numpy as np
from .layer import Layer

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, input_data):
        if self.is_training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            self.output = input_data * self.mask
        else:
            self.output = input_data
        return self.output

    def backward(self, grad_output):
        return grad_output * self.mask
