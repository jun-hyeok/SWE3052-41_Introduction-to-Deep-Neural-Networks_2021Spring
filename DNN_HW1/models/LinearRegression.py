import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W
        print(f"x.shape {x.shape}, y.shape{y.shape}")
        n_batch = int(x.shape[0]/batch_size)

        for i in range(epochs):    
            final_loss = 0
            dw = np.zeros_like(self.W)
            for j in range(n_batch):
                batch_start = j * batch_size
                batch_end = batch_start+batch_size
                x_batched = x[batch_start:batch_end]
                y_batched = y[batch_start:batch_end]

                y_predicted = self.forward(x_batched)
                error = y_predicted - y_batched
                final_loss += np.sum(np.square(error)) / batch_size
                dw =  2 * np.sum(x_batched * error, axis=0).reshape(self.num_features, 1) 
                dw /= batch_size
                w = optim.update(w, dw, lr)
                self.W = w
            final_loss /= n_batch
        print(f"loss {final_loss}, batch_size {batch_size}, epoch {epochs}")    

        # ============================================================
        return final_loss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)

        # ============================================================
        return y_predicted
