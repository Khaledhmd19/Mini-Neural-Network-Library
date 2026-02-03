import numpy as np
from .optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layer):
        if layer not in self.cache:
            self.cache[layer] = {}
            for param_name in layer.params:
                self.cache[layer][param_name] = np.zeros_like(layer.params[param_name])

        for param_name in layer.params:
            if param_name in layer.grads:
                grad = layer.grads[param_name]
                self.cache[layer][param_name] += grad ** 2
                
                denom = np.sqrt(self.cache[layer][param_name]) + self.epsilon
                layer.params[param_name] -= (self.learning_rate / denom) * grad
