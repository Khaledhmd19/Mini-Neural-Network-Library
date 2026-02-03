from layer.layer import Layer


class Loss(Layer):

    def forward(self, prediction, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
