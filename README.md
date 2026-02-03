# HammoudMiniNN

A lightweight, educational neural network library built from scratch in Python. This project implements core deep learning concepts including forward/backward propagation, various activation functions, loss functions, and optimization algorithms.

## Features

- **Multiple Activation Functions**: Linear, ReLU, Sigmoid, Tanh
- **Loss Functions**: Mean Squared Error, Softmax Cross Entropy
- **Optimization Algorithms**: SGD, Momentum, Adagrad, Adam
- **Layer Types**: Dense layers, Dropout, Batch Normalization
- **Training & Tuning**: Built-in trainer and hyperparameter tuning utilities
- **Example Use Cases**: MNIST digit classification, XOR problem

## Project Structure

```
HammoudMiniNN/
├── network.py              # Main neural network class
├── trainer.py              # Training loop and utilities
├── tuning.py               # Hyperparameter tuning tools
├── activation/             # Activation function implementations
│   ├── linear.py
│   ├── relu.py
│   ├── sigmoid.py
│   └── tanh.py
├── layer/                  # Layer implementations
│   ├── dense.py
│   ├── dropout.py
│   └── batch_normalization.py
├── loss/                   # Loss function implementations
│   ├── mean_squared_error.py
│   └── softmax_cross_entropy.py
├── optimizers/             # Optimizer implementations
│   ├── adam.py
│   ├── adagrad.py
│   ├── momentum.py
│   └── sgd.py
└── examples/               # Example implementations
    ├── mnist_digits.py
    └── xor.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HammoudMiniNN.git
cd HammoudMiniNN
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. No external dependencies required! This is a pure Python implementation.

## Quick Start

### Create a Simple Neural Network

```python
from network import Network
from layer import Dense
from activation import ReLU, Sigmoid
from loss import MeanSquaredError
from optimizers import Adam

# Initialize network
network = Network()

# Add layers
network.add(Dense(input_size=784, units=128))
network.add(ReLU())
network.add(Dense(input_size=128, units=64))
network.add(ReLU())
network.add(Dense(input_size=64, units=10))
network.add(Sigmoid())

# Compile network
network.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.001)
)

# Train network
network.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predictions = network.predict(X_test)
```

### Examples

Run the included examples:

```bash
# MNIST digit classification
python examples/mnist_digits.py

# XOR problem
python examples/xor.py
```

## Activation Functions

- **Linear**: Identity function, no transformation
- **ReLU**: Rectified Linear Unit - max(0, x)
- **Sigmoid**: Smooth S-shaped function, output in (0, 1)
- **Tanh**: Hyperbolic tangent, output in (-1, 1)

## Optimizers

- **SGD**: Stochastic Gradient Descent
- **Momentum**: SGD with momentum
- **Adagrad**: Adaptive learning rates per parameter
- **Adam**: Adaptive moment estimation (recommended)

## Loss Functions

- **Mean Squared Error (MSE)**: For regression tasks
- **Softmax Cross Entropy**: For multi-class classification

## Performance Tips

- Start with smaller networks and increase complexity
- Use Batch Normalization for deeper networks
- Apply Dropout for regularization to prevent overfitting
- Use Adam optimizer for most tasks - it's very robust
- Normalize your input data before training

## Learning Resources

This implementation is designed for educational purposes. Key concepts covered:

- Backpropagation algorithm
- Gradient descent optimization
- Neural network architecture design
- Regularization techniques

## Contributing

Feel free to fork, modify, and improve this project! Some ideas:

- Add more activation functions (ELU, SELU, etc.)
- Implement Convolutional layers
- Add Recurrent layer support
- Improve documentation and examples
