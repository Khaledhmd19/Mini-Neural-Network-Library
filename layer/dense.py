import numpy as np
from .layer import Layer

class Dense(Layer):
    def __init__(self, input_dim, output_dim, init_method='xavier'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if init_method == 'xavier':
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.params['W'] = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif init_method == 'he':
            std = np.sqrt(2 / input_dim)
            self.params['W'] = np.random.randn(input_dim, output_dim) * std
        else:
            self.params['W'] = np.random.randn(input_dim, output_dim) * 0.01
            
        self.params['b'] = np.zeros((1, output_dim))
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.params['W']) + self.params['b']
        return self.output

    def backward(self, grad_output):
        self.grads['W'] = np.dot(self.input.T, grad_output)
        
        self.grads['b'] = np.sum(grad_output, axis=0, keepdims=True)
        
        grad_input = np.dot(grad_output, self.params['W'].T)
        
        return grad_input
