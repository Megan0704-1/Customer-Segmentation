import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

parser = argparse.ArgumentParser(description="Train a simple CNN on MNIST")

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

parser.add_argument(
    "--kernel-size",
    type=int,
    default=3,
    help="Kernel size for all convolution layers (default: 3)",
)

parser.add_argument(
    "--filter1",
    type=int,
    default=16,
    help="Number of output channels in the 1st convolution layers (default: 16)",
)
parser.add_argument(
    "--filter2",
    type=int,
    default=32,
    help="Number of output channels in the 2nd convolution layers (default: 32)",
)
parser.add_argument(
    "--filter3",
    type=int,
    default=16,
    help="Number of output channels in the 3rd convolution layers (default: 16)",
)

args = parser.parse_args()


n_epochs = args.epochs  # Number of Epochs for training
batch_size = args.batch_size  # Batch size for training and testing
log_interval = (
    args.log_interval
)  # This variable manages how frequently do you want to print the training loss

kernel_size = args.kernel_size
f1 = args.filter1
f2 = args.filter2
f3 = args.filter3

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


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # MNIST has 1 input channel (grayscale)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=f1, kernel_size=kernel_size, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=f1, out_channels=f2, kernel_size=kernel_size, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=f2, out_channels=f3, kernel_size=kernel_size, padding=1
        )

        # pooling after every conv reduces spatial size
        # e.g., 28*28 => 14*14
        # final feature map has size [batch, f3, 3, 3]
        self.fc = nn.Linear(f3 * 9, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28*28 -> 14*14

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14*14 -> 7*7

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # 7*7 -> 3*3 (floor)

        x = x.view(x.size(0), -1)  # flatten

        logits = self.fc(x)
        return logits


network = SimpleCNN()
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
