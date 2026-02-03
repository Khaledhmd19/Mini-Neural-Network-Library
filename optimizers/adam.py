import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        self.t += 1
        if layer not in self.m:
            self.m[layer] = {}
            self.v[layer] = {}
            for param_name in layer.params:
                self.m[layer][param_name] = np.zeros_like(layer.params[param_name])
                self.v[layer][param_name] = np.zeros_like(layer.params[param_name])

        for param_name in layer.params:
            if param_name in layer.grads:
                grad = layer.grads[param_name]
                
                self.m[layer][param_name] = self.beta1 * self.m[layer][param_name] + (1 - self.beta1) * grad
                self.v[layer][param_name] = self.beta2 * self.v[layer][param_name] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[layer][param_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[layer][param_name] / (1 - self.beta2 ** self.t)
                
                layer.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
