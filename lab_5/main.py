import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from scipy.optimize import differential_evolution
import torchvision
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')


device = torch.device("cuda")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


def train_eval_model(params):
    lr, dropout = params

    class CNN(nn.Module):
        def __init__(self, num_classes=100):
            super(CNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 4 * 4, num_classes),
                nn.LogSoftmax(dim=1)
            )

        def forward(self, x):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

    model = CNN().to(device)
    # summary(model, input_size=(3, 32, 32))
    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = 1
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        val_accuracies.append(acc)
    print(f"Точність {100 * correct / total:.2f}%")
    return -acc


param_bounds = [
    (1e-8, 1e-2),
    (0.1, 0.7),
]

# differential_evolution(train_eval_model, param_bounds, maxiter=1, popsize=2, disp=True)


lr_values = np.logspace(-5, -2, 10)
dropout_values = np.linspace(0.1, 0.7, 10)

Z = np.zeros((len(dropout_values), len(lr_values)))

for i, dropout in enumerate(dropout_values):
    for j, lr in enumerate(lr_values):
        acc = -train_eval_model([lr, dropout])
        Z[i, j] = acc

LR, DO = np.meshgrid(lr_values, dropout_values)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(np.log10(LR), DO, Z, cmap='viridis')

ax.set_xlabel('Learning rate')
ax.set_ylabel('Dropout')
ax.set_zlabel('Точність')


fig.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
