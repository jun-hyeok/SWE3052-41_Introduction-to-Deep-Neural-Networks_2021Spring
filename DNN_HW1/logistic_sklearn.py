from utils import _initialize, optimizer
import sklearn
from sklearn.linear_model import LogisticRegression

# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters
# ========================= EDIT HERE ========================
# DATA
DATA_NAME = 'Titanic'
DATA_NAME = 'Digit'

# HYPERPARAMETERS
num_epochs = 300
# ============================================================

assert DATA_NAME in ['Titanic', 'Digit','Basic_coordinates']

# Load dataset, model and evaluation metric
train_data, test_data, _, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)
ACC = 0.0
# ========================= EDIT HERE ========================
# Make model & optimizer
model = LogisticRegression(max_iter=num_epochs)
optim = optimizer("SGD")

# TRAIN
res = model.fit(train_x, train_y)

# EVALUATION
test_x, test_y = test_data
predict_y = model.predict(test_x).reshape(test_x.shape[0], 1)
ACC = metric(predict_y, test_y)
# ============================================================

print('ACC on Test Data : %.2f ' % ACC)
