import numpy as np
import time

class Trainer:
    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.model.set_loss(loss_function)
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    def fit(self, X_train, y_train, epochs=10, batch_size=32, X_val=None, y_val=None, verbose=True):
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            start_time = time.time()
            self.model.set_training_mode(True)
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices] if y_train.ndim > 1 else y_train[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size] if y_train.ndim > 1 else y_train[i:i+batch_size]
                
                loss = self.model.train_step(X_batch, y_batch, self.optimizer)
                epoch_loss += loss
                
                pred = self.model.predict(X_batch)
                if y_batch.ndim > 1:
                    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_batch, axis=1))
                else:
                    acc = np.mean(np.argmax(pred, axis=1) == y_batch)
                epoch_acc += acc
                n_batches += 1
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            self.history['loss'].append(epoch_loss)
            self.history['acc'].append(epoch_acc)
            
            val_loss = 0
            val_acc = 0
            if X_val is not None and y_val is not None:
                self.model.set_training_mode(False)
                val_pred = self.model.predict(X_val)
                val_loss = self.model.loss_function.forward(val_pred, y_val)
                
                if y_val.ndim > 1:
                    val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                else:
                    val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
                    
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}", end="")
                if X_val is not None:
                    print(f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print("")
                    
        return self.history
