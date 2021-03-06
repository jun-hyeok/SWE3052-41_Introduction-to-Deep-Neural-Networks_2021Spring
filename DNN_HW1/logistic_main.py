import numpy as np
from utils import _initialize, optimizer

np.random.seed(428)

# ========================= EDIT HERE ========================
# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters
# 3. Choose Optimizer : SGD

# DATA
DATA_NAME = 'Titanic'
# DATA_NAME = 'Digit'

# HYPERPARAMETERS
batch_size = 10
# batch_size = 779 # for full batch
num_epochs = 300
# num_epochs = 300 # for trial
learning_rate = 0.0005
# learning_rate = 0.4 # for trial

# ============================================================
epsilon = 0.01 # not for SGD
gamma = 0.9 # not for SGD

# OPTIMIZER
OPTIMIZER = 'SGD'

assert DATA_NAME in ['Titanic', 'Digit', 'Basic_coordinates']
assert OPTIMIZER in ['SGD']

# Load dataset, model and evaluation metric
train_data, test_data, logistic_regression, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)

# Make model & optimizer
model = logistic_regression(num_features)
optim = optimizer(OPTIMIZER, gamma=gamma, epsilon=epsilon)

# TRAIN
loss = model.train(train_x, train_y, num_epochs, batch_size, learning_rate, optim)
print('Training Loss at the last epoch: %.2f' % loss)

# EVALUATION
test_x, test_y = test_data
pred = model.forward(test_x)
ACC = metric(pred, test_y)

print(OPTIMIZER, ' ACC on Test Data : %.3f' % ACC)
