from .optimizer import Optimizer

class SGD(Optimizer):
    def update(self, layer):
        for param_name in layer.params:
            if param_name in layer.grads:
                layer.params[param_name] -= self.learning_rate * layer.grads[param_name]
