from .activation import Activation

class Linear(Activation):
    def forward(self, input_data):
        self.input = input_data
        self.output = input_data
        return self.output

    def backward(self, grad_output):
        return grad_output
