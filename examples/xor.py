import numpy as np

from layer.dense import Dense
from activation.sigmoid import Sigmoid
from loss.mean_squared_error import MeanSquaredError
from optimizers.sgd import SGD
from network import NeuralNetwork
from trainer import Trainer

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Build model
model = NeuralNetwork()
model.add(Dense(2, 2))  # hidden layer with 2 neurons
model.add(Sigmoid())
model.add(Dense(2, 1))  # output layer with 1 neuron
model.add(Sigmoid())  # output activation

# Loss and optimizer
loss_fn = MeanSquaredError()
optimizer = SGD(learning_rate=0.5)

# Trainer
trainer = Trainer(model, optimizer, loss_fn)
history = trainer.fit(X, y, epochs=5000, batch_size=4)

# Predictions
preds = model.predict(X)
preds_binary = (preds > 0.5).astype(int)
print("Predictions:\n", preds_binary)


# python -m examples.xor