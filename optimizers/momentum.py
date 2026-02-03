import numpy as np
from .optimizer import Optimizer

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer):
        if layer not in self.velocities:
            self.velocities[layer] = {}
            for param_name in layer.params:
                self.velocities[layer][param_name] = np.zeros_like(layer.params[param_name])

        for param_name in layer.params:
            if param_name in layer.grads:
                v = self.velocities[layer][param_name]
                v = self.momentum * v - self.learning_rate * layer.grads[param_name]
                self.velocities[layer][param_name] = v
                layer.params[param_name] += v
