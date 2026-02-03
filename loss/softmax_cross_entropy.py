import numpy as np
from .loss import Loss

class SoftmaxCrossEntropy(Loss):
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        
        exps = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        self.softmax_output = exps / np.sum(exps, axis=1, keepdims=True)
        
        epsilon = 1e-12
        self.softmax_output = np.clip(self.softmax_output, epsilon, 1. - epsilon)
        
        if target.ndim > 1:
            loss = -np.sum(target * np.log(self.softmax_output)) / prediction.shape[0]
        else:
            m = target.shape[0]
            log_likelihood = -np.log(self.softmax_output[range(m), target])
            loss = np.sum(log_likelihood) / m
            
        return loss

    def backward(self, grad_output=None):
        if self.target.ndim > 1:
            grad = (self.softmax_output - self.target) / self.prediction.shape[0]
        else:
            m = self.target.shape[0]
            grad = self.softmax_output.copy()
            grad[range(m), self.target] -= 1
            grad = grad / m
            
        return grad
