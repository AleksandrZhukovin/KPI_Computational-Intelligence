import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torchvision import models


def train_model():
    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder("dataset/train", transform=transform)
    val_ds = datasets.ImageFolder("dataset/valid", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

    class_names = train_ds.classes
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "plant_disease_model.pth")
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_names))
