import numpy as np
from tensorflow.keras.datasets import mnist

from layer import Dense, Dropout, BatchNormalization
from activation import ReLU
from loss import SoftmaxCrossEntropy
from optimizers import Adam
from network import NeuralNetwork
from trainer import Trainer

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

y_train_oh = np.eye(10)[y_train]
y_test_oh = np.eye(10)[y_test]

# Build model
model = NeuralNetwork()
model.add(Dense(784, 128))
model.add(ReLU())
model.add(BatchNormalization(128))
model.add(Dropout(0.2))
model.add(Dense(128, 64))
model.add(ReLU())
model.add(Dense(64, 10))   # output layer

# Loss and optimizer
loss_fn = SoftmaxCrossEntropy()
optimizer = Adam(learning_rate=0.001)

# Trainer
trainer = Trainer(model, optimizer, loss_fn)
history = trainer.fit(X_train, y_train_oh, epochs=10, batch_size=64, X_val=X_test, y_val=y_test_oh)

# Evaluate
preds = np.argmax(model.predict(X_test), axis=1)
accuracy = np.mean(preds == y_test)
print("Test Accuracy:", accuracy)