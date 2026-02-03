import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_function):
        self.loss_function = loss_function

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def forward(self, input_data):
        return self.predict(input_data)

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train_step(self, X, y, optimizer):
        prediction = self.predict(X)
        
        loss = self.loss_function.forward(prediction, y)
        
        grad_loss = self.loss_function.backward()
        self.backward(grad_loss)
        
        for layer in self.layers:
            optimizer.update(layer)
            
        return loss

    def set_training_mode(self, is_training):
        for layer in self.layers:
            layer.set_training_mode(is_training)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
