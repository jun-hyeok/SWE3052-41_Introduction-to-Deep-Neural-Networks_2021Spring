import numpy as np
import pandas as pd

from dataset import SSL_Dataset
from model import CNN
from utils import save_prediction, plot_accuracy

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from time import time

"""
Please use this code as a guideline. 
Feel free to create your own code for training, testing, ... etc.
But for creating "submission.csv" file, utilizing this code is highly recommended.
"""


class Trainer:
    def __init__(
        self,
        model,
        device,
        weight_path,
        model_name,
        patience,
        momentum,
        weight_decay,
        learning_rate,
        num_epoch,
        print_every,
    ):

        self.patience = patience
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.print_every = print_every

        self.best_acc = 0
        self.best_epoch = 0
        self.crnt_epoch = 0
        self.endure = 0
        self.stop_flag = False
        self.num_class = 10
        self.device = device
        self.weight_path = weight_path
        self.model_name = model_name

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.train_acc = []
        self.valid_acc = []

    # test
    def _test(self, mode, data_loader):

        test_preds = []
        self.model.eval()
        correct = 0
        total = 0

        if mode == "Valid":
            with torch.no_grad():
                for batch_data in data_loader:
                    batch_x, batch_y = batch_data
                    inputs, targets = batch_x.to(self.device), batch_y.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    total += targets.size(0)
                    correct += predicted.eq(targets).cpu().sum().item()
                    if self.device == "cuda":
                        test_preds += predicted.detach().cpu().numpy().tolist()
                    else:
                        test_preds += predicted.detach().numpy().tolist()

            total_acc = correct / total

            print(
                "| \033[31m%s Epoch #%d\t Accuracy: %.2f%%\033[0m"
                % (mode, self.crnt_epoch + 1, 100.0 * total_acc)
            )
            if self.crnt_epoch % self.print_every == 0:
                self.valid_acc.append(total_acc)
            if self.best_acc < total_acc:
                print(
                    "| \033[32mBest Accuracy updated (%.2f => %.2f)\033[0m\n"
                    % (100.0 * self.best_acc, 100.0 * total_acc)
                )
                self.best_acc = total_acc
                self.best_epoch = self.crnt_epoch
                self.endure = 0
                # Save best model
                torch.save(self.model.state_dict(), self.weight_path + self.model_name)
            else:
                self.endure += 1
                print(f"| Endure {self.endure} out of {self.patience}\n")
                if self.endure >= self.patience:
                    print("Early stop triggered...!")
                    self.stop_flag = True

        if mode == "Test":
            print("Predicting Starts...")
            with torch.no_grad():
                for batch_data in data_loader:
                    batch_x = batch_data
                    try:
                        inputs = batch_x.to(self.device)
                    except:
                        inputs = batch_x[0].to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    if self.device == "cuda":
                        test_preds += predicted.detach().cpu().numpy().tolist()
                    else:
                        test_preds += predicted.detach().numpy().tolist()

            return test_preds, self.crnt_epoch, self.train_acc, self.valid_acc

    # train
    def _train(self, labeled_trainloader, labeled_validloader):
        self.model.train()
        print("Training Starts...")
        total = 0
        correct = 0
        for epoch in range(self.num_epoch):
            self.crnt_epoch = epoch
            for batch_data in labeled_trainloader:
                batch_x, batch_y = batch_data
                batch_size = batch_x.size(0)
                batch_y = torch.zeros(batch_size, self.num_class).scatter_(
                    1, batch_y.view(-1, 1), 1
                )
                inputs_l, targets_l = batch_x.to(self.device), batch_y.long().to(
                    self.device
                )

                outputs = self.model(inputs_l)
                loss = self.loss_fn(outputs, torch.argmax(targets_l, dim=-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Epoch: {epoch+1}, Loss:{loss:.4f}")

                _, predicted = torch.max(outputs, 1)

                total += targets_l.size(0)
                correct += (
                    predicted.eq(torch.argmax(targets_l, dim=-1)).cpu().sum().item()
                )

            total_acc = correct / total
            if self.crnt_epoch % self.print_every == 0:
                self.train_acc.append(total_acc)
            print(
                "\n| \033[31mTrain Epoch #%d\t Accuracy: %.2f%%\033[0m"
                % (self.crnt_epoch + 1, 100.0 * total_acc)
            )
            pred = self._test("Valid", labeled_validloader)
            if self.stop_flag:
                break
        print("Training Finished...!!")


def main():

    #################### EDIT HERE ####################
    """
    You can change any values of hyper-parameter below.
    *test_only: If this parameter is True, you can just test with a model that already exists without training step.
    (for your time saving..!)
    """
    random_seed = 1

    patience = 10
    momentum = 0.9
    weight_decay = 5e-4
    learning_rate = 0.003

    num_epoch = 100
    print_every = 1
    train_batch = 512
    test_batch = 1000
    valid_ratio = 0.1
    psuedo_ratio = 0.5

    model_name = "my_model.p"
    train_data_mode = "labeled_train"

    test_only = False
    ###################################################

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
    else:
        device = "cpu"

    weight_path = "./best_model/"

    transform = transforms.Compose(
        [transforms.Resize(size=(32, 32)), transforms.ToTensor()]
    )

    train_examples = SSL_Dataset(root="../", transform=transform, mode=train_data_mode)
    num_train = len(train_examples)
    num_valid = int(num_train * valid_ratio)
    train_labeled_dataset, valid_labeled_dataset = torch.utils.data.random_split(
        train_examples, [num_train - num_valid, num_valid]
    )
    test_labeled_dataset = SSL_Dataset(root="../", transform=transform, mode="test")
    test_idx = test_labeled_dataset.idx

    labeled_trainloader = DataLoader(
        train_labeled_dataset, batch_size=train_batch, shuffle=True
    )
    labeled_validloader = DataLoader(
        valid_labeled_dataset, batch_size=test_batch, shuffle=False
    )
    labeled_testloader = DataLoader(
        test_labeled_dataset, batch_size=test_batch, shuffle=False
    )

    #################### EDIT HERE ####################
    train_unlabeled_dataset = SSL_Dataset(
        root="../", transform=transform, mode="unlabeled_train"
    )
    unlabeled_trainloader = DataLoader(
        train_unlabeled_dataset, batch_size=test_batch, shuffle=False
    )
    """
    If you want to try with ResNet18 model :
    """
    model = models.resnet18().to(device)
    model.fc = nn.Linear(model.fc.in_features, 10).to(device)

    """
    If you want to try with custom model (CNN in model.py) :
    """
    # model = CNN().to(device)

    """
    Or you can try any model! :
    """
    ###################################################

    trainer = Trainer(
        model,
        device,
        weight_path,
        model_name,
        patience,
        momentum,
        weight_decay,
        learning_rate,
        num_epoch,
        print_every,
    )

    if test_only == False:
        print(
            f"# Train data: {len(train_labeled_dataset)}, # Valid data: {len(valid_labeled_dataset)}"
        )
        train_start = time()
        trainer._train(labeled_trainloader, labeled_validloader)
        train_elapsed = time() - train_start
        print("Train Time: %.4f\n" % train_elapsed)
        #################### EDIT HERE ####################
        psuedo_label, *_ = trainer._test("Test", unlabeled_trainloader)
        train_psuedo_labeled_dataset = SSL_Dataset(
            root="../", transform=transform, mode=train_data_mode
        )
        train_psuedo_labeled_dataset.data = train_unlabeled_dataset.data
        train_psuedo_labeled_dataset.targets = torch.LongTensor(psuedo_label)

        num_train = len(train_examples)
        num_unlabel = len(train_psuedo_labeled_dataset)
        num_psuedo = int(num_train * psuedo_ratio)
        num_valid = int(num_train * valid_ratio)

        train_labeled_dataset, _ = torch.utils.data.random_split(
            train_examples, [num_train - num_psuedo, num_psuedo]
        )
        _, train_psuedo_labeled_dataset = torch.utils.data.random_split(
            train_psuedo_labeled_dataset, [num_unlabel - num_psuedo, num_psuedo]
        )

        (
            train_semi_labeled_dataset,
            valid_semi_labeled_dataset,
        ) = torch.utils.data.random_split(
            train_labeled_dataset + train_psuedo_labeled_dataset,
            [num_train - num_valid, num_valid],
        )

        semi_labeled_trainloader = DataLoader(
            train_semi_labeled_dataset, batch_size=train_batch, shuffle=True
        )
        semi_labeled_validloader = DataLoader(
            valid_semi_labeled_dataset, batch_size=test_batch, shuffle=False
        )
        train_start = time()
        trainer._train(semi_labeled_trainloader, semi_labeled_validloader)
        train_elapsed = time() - train_start
        print("Train Time: %.4f\n" % train_elapsed)
        ###################################################

    print(f"# Test data: {len(test_labeled_dataset)}")
    model.load_state_dict(torch.load(weight_path + model_name))
    pred, num_epochs, train_acc, valid_acc = trainer._test("Test", labeled_testloader)
    save_prediction(weight_path, pred, test_idx)
    plot_accuracy(print_every, weight_path, num_epochs, train_acc, valid_acc)


if __name__ == "__main__":
    main()