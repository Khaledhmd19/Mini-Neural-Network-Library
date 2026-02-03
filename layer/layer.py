class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input = None
        self.output = None
        self.is_training = True

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
    
    def set_training_mode(self, is_training):
        self.is_training = is_training
