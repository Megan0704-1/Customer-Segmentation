import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST")

parser.add_argument(
    "--hidden-neurons",
    type=int,
    default=128,
    help="Number of hidden neurons in the Simple MLP model"
)

parser.add_argument(
    "--layers",
    type=int,
    default=3,
    help="Number of hidden layers in the Simple MLP model"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for training and testing (default: 64)",
)

parser.add_argument(
    "--epochs", type=int, default=1, help="Number of epochs to train (default: 1)"
)

parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="How many batches to wait before logging training status (default: 100)",
)

args = parser.parse_args()

hidden = args.hidden_neurons
n_epochs = args.epochs  # Number of Epochs for training
batch_size = (
    args.batch_size
)  # Batch size for training and testing TODO: Modify this variable to change batch size
log_interval = (
    args.log_interval
)  # This variable manages how frequently do you want to print the training loss

####################################################################
# Avoid changing the below parameters
learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

####################################################################
# Train loader and test loader for the MNIST dataset
# This part of the code will download the MNIST dataset in the
# same directory as this script. It will normalize the dataset and batch
# it into the batch size specified by batch_size var.

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

####################################################################
# TODO: Define your model here
# See the example MLP linked in the lab document for help.


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 128)
        self.fc_add = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        if args.layers == 4:
            x = F.relu(self.fc_add(x))
            x = F.relu(self.fc2(x))
        elif args.layers == 3:
            x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))
        logits = self.softmax(x)
        return logits


network = SimpleMLP()
# Using the SGD (Stochastic Gradient Descent) optimizer
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []

####################################################################
# Train and test methods for training the model


def train(epoch):
    network.train()
    total_training_time = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_end_time = time.time()
        total_training_time += batch_end_time - batch_start_time
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
    return total_training_time


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


####################################################################
# Train the model for given epochs

total_time = 0
for epoch in range(1, n_epochs + 1):
    time_per_epoch = train(epoch)
    total_time = total_time + time_per_epoch
    test()

print("Total Training time: {}".format(total_time))

####################################################################
# Single inference

with torch.no_grad():
    test_iterator = iter(test_loader)
    data, target = next(test_iterator)
    single_batch_start = time.time()
    # Run single inference for 1000 times to avoid measurement overheads
    for i in range(0, 1000):
        output = network(data)
    single_batch_end = time.time()

    single_batch_inf_time = (single_batch_end - single_batch_start) / 1000
    print(
        "Single Batch Inference time is {} seconds for a batch size of {}".format(
            single_batch_inf_time, test_loader.batch_size
        )
    )
