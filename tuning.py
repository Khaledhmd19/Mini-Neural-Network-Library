import numpy as np
import itertools
from .network import NeuralNetwork
from .layer import Dense, Dropout, BatchNormalization
from .activation import ReLU, Sigmoid, Tanh, Linear
from .loss import MeanSquaredError, SoftmaxCrossEntropy
from .optimizers import SGD, Momentum, Adagrad, Adam
from .trainer import Trainer

class HyperparameterTuning:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.results = []

    def grid_search(self, param_grid):
        keys = param_grid.keys()
        combinations = itertools.product(*param_grid.values())
        
        best_acc = 0
        best_config = None
        best_model = None
        
        for combo in combinations:
            config = dict(zip(keys, combo))
            print(f"Testing configuration: {config}")
            
            model = self._build_model(config)
            optimizer = self._get_optimizer(config)
            loss_fn = self._get_loss(config)
            
            trainer = Trainer(model, optimizer, loss_fn)
            history = trainer.fit(self.X_train, self.y_train, 
                                  epochs=config.get('epochs', 10), 
                                  batch_size=config.get('batch_size', 32),
                                  X_val=self.X_val, y_val=self.y_val,
                                  verbose=False)
            
            val_acc = history['val_acc'][-1]
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            self.results.append({'config': config, 'val_acc': val_acc, 'history': history})
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_config = config
                best_model = model
                
        print(f"\nBest Configuration: {best_config}")
        print(f"Best Validation Accuracy: {best_acc:.4f}")
        return best_config, best_model

    def _build_model(self, config):
        model = NeuralNetwork()
        input_dim = self.X_train.shape[1]
        
        layers_config = config.get('layers', [64, 32])
        activation = config.get('activation', 'relu')
        dropout_rate = config.get('dropout', 0.0)
        use_batchnorm = config.get('batchnorm', False)
        
        for units in layers_config:
            model.add(Dense(input_dim, units))
            if use_batchnorm:
                model.add(BatchNormalization(units))
            
            if activation == 'relu':
                model.add(ReLU())
            elif activation == 'sigmoid':
                model.add(Sigmoid())
            elif activation == 'tanh':
                model.add(Tanh())
                
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
                
            input_dim = units
            
        output_dim = self.y_train.shape[1] if self.y_train.ndim > 1 else len(np.unique(self.y_train))
        model.add(Dense(input_dim, output_dim))
        
        return model

    def _get_optimizer(self, config):
        opt_name = config.get('optimizer', 'adam')
        lr = config.get('learning_rate', 0.01)
        
        if opt_name == 'sgd':
            return SGD(learning_rate=lr)
        elif opt_name == 'momentum':
            return Momentum(learning_rate=lr)
        elif opt_name == 'adagrad':
            return Adagrad(learning_rate=lr)
        elif opt_name == 'adam':
            return Adam(learning_rate=lr)
        return SGD(learning_rate=lr)

    def _get_loss(self, config):
        loss_name = config.get('loss', 'softmax_crossentropy')
        if loss_name == 'mse':
            return MeanSquaredError()
        return SoftmaxCrossEntropy()
