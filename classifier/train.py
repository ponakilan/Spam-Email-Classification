import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

from metrics import accuracy


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0

    def early_stop(self, validation_acc, model):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
            file_name = "weights/EmailClassifier.pt"
            torch.save(model.state_dict(), file_name)
            print(f"Model saved to {file_name}")
        elif validation_acc < (self.max_validation_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, train_dataloader, valid_dataloader, epoch, device, show_plot=False):
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    early_stopper = EarlyStopper(patience=2, min_delta=0)

    history = {
        "loss": [],
        "train_acc": [],
        "valid_acc": [],
    }

    for i in range(epoch):
        print(f"Epoch {i} has started.")
        model.train()
        loss = None
        for j, data in enumerate(train_dataloader):
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_func(outputs, Y.reshape(-1, 1).to(torch.float32))
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        train_acc = accuracy(train_dataloader, model, device)
        valid_acc = accuracy(valid_dataloader, model, device)
        history["loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        print(f'Loss: {loss.item()}')
        print(f'Train Accuracy: {train_acc}')
        print(f'Valid Accuracy: {valid_acc}\n')
        if early_stopper.early_stop(valid_acc, model):
            print("Early stopper triggered. Interrupting training...")
            break
    print("Training Complete.")

    if show_plot:
        plt.title("Training loss")
        plt.plot(history["loss"])
        plt.show()

        plt.title("Accuracy")
        plt.plot(history["train_acc"])
        plt.plot(history["valid_acc"])
        plt.legend(["Training accuracy", "Validation accuracy"])
        plt.show()

    return history
