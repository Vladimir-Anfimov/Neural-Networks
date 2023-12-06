import pickle
import gzip
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device, "\n")


class MNISTDataset(Dataset):
    def __init__(self, train=True):
        with gzip.open("mnist.pkl.gz", "rb") as fd:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(fd, encoding="latin")

        if train:
            self.inputs = torch.from_numpy(np.concatenate((train_x, test_x)))
            self.labels = torch.from_numpy(np.concatenate((train_y, test_y)))
        else:
            self.inputs = torch.from_numpy(valid_x)
            self.labels = torch.from_numpy(valid_y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            # nn.Dropout(0.5) Nu ajuta la accuracy
            nn.Linear(128, 10),
            # nn.Softmax(dim=1) Am folosit CrossEntropyLoss care are deja Softmax inclus
        )

    def forward(self, x):
        return self.model(x)


def test(model, test_loader):
    correct = 0
    tp = 0
    fn = 0
    fp = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            for i in range(len(inputs)):
                if predicted[i] == 1 and labels[i] == 1:
                    tp += 1
                elif predicted[i] == 0 and labels[i] == 1:
                    fn += 1
                elif predicted[i] == 1 and labels[i] == 0:
                    fp += 1

    accuracy = correct / len(test_loader.dataset) * 100
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Accuracy: ", round(accuracy, 4))
    print("Recall: ", round(recall, 4))
    print("Precision: ", round(precision, 4))
    print("F1-score: ", round(f1_score, 4))
    print()


def train(model, train_loader, epochs=10):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    time_start = time.time()
    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    print(f"Training time: {time.time() - time_start}s")



model = Net().to(device)
train_loader = DataLoader(MNISTDataset(), batch_size=128, shuffle=True)

train(model, train_loader, epochs=10)
test(model, train_loader)

test_loader = DataLoader(MNISTDataset(train=False), batch_size=128, shuffle=True)
test(model, test_loader)