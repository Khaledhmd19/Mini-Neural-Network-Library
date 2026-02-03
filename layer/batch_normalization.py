import numpy as np
from .layer import Layer

class BatchNormalization(Layer):
    def __init__(self, input_dim, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.params['gamma'] = np.ones((1, input_dim))
        self.params['beta'] = np.zeros((1, input_dim))
        
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        
        self.cache = None

    def forward(self, input_data):
        if self.is_training:
            batch_mean = np.mean(input_data, axis=0, keepdims=True)
            batch_var = np.var(input_data, axis=0, keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            x_centered = input_data - batch_mean
            std_inv = 1. / np.sqrt(batch_var + self.epsilon)
            x_norm = x_centered * std_inv
            
            self.cache = (x_centered, std_inv, x_norm)
            self.output = self.params['gamma'] * x_norm + self.params['beta']
        else:
            x_norm = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.params['gamma'] * x_norm + self.params['beta']
            
        return self.output

    def backward(self, grad_output):
        x_centered, std_inv, x_norm = self.cache
        N = grad_output.shape[0]
        
        self.grads['gamma'] = np.sum(grad_output * x_norm, axis=0, keepdims=True)
        self.grads['beta'] = np.sum(grad_output, axis=0, keepdims=True)
        
        dx_norm = grad_output * self.params['gamma']
        dvar = np.sum(dx_norm * x_centered * -0.5 * std_inv**3, axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2. * x_centered, axis=0, keepdims=True)
        
        dx = dx_norm * std_inv + dvar * 2 * x_centered / N + dmean / N
        return dx
