import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 1e-7
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W
        print(f"x.shape {x.shape}, y.shape{y.shape}")
        n_batch = int(x.shape[0]/batch_size)
        for i in range(epochs):
            loss = 0
            dw = np.zeros_like(self.W)
            for j in range(n_batch):
                batch_start = j * batch_size
                batch_end = batch_start+batch_size
                x_batched = x[batch_start:batch_end]
                y_batched = y[batch_start:batch_end]
                y_predicted = self.forward(x_batched)
                neg_cost = y_batched * np.log(y_predicted+epsilon) + (1 - y_batched) * np.log(1 - y_predicted+epsilon)
                loss += -np.sum(neg_cost) / batch_size
                error = y_predicted - y_batched
                dw = np.sum(x_batched * error, axis=0).reshape(self.num_features, 1)
                dw /= batch_size
                w = optim.update(w, dw, lr)
                self.W = w
            loss /= n_batch
        print(f"loss {loss}, batch_size {batch_size}, epoch {epochs}")

        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        y_predicted = LogisticRegression._sigmoid(self, y_predicted)
        lt_threshold = y_predicted < threshold
        y_predicted[lt_threshold] = 0
        y_predicted[np.invert(lt_threshold)] = 1

        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 + np.exp(-x))

        # ============================================================
        return sigmoid
