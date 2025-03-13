#Train target model with good generalizability on CIfAr-10 dataset 

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torchvision.models as models
from torchvision.models import resnet18



BASE_DIR = "/home/lab24inference/amelie/target_model/"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "target_model.pth")
EPOCH_TRACKER_PATH = os.path.join(BASE_DIR, "epoch_tracker.json")
TRAINING_PLOT_PATH = os.path.join(BASE_DIR, "training_plot.png")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Apply Augmentation 
def load_data():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(BASE_DIR, "data"), train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16)  # Batch-Größe erhöht
    testset = torchvision.datasets.CIFAR10(root=os.path.join(BASE_DIR, "data"), train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=16)  # Batch-Größe erhöht
    return trainloader, testloader

trainloader, testloader = load_data()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the target model
class TargetModel(nn.Module):
    def __init__(self):
        super(TargetModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4 (Zusätzliches Pooling)
        x = x.view(x.size(0), -1)  # Dynamisches Flattening
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x


model = TargetModel().to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)  # Lernrate nach 50 Epochen anpassen

# Load previously saved model and training progress
def load_checkpoint():
    if os.path.exists(MODEL_SAVE_PATH):
        print("Load saved model...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    if os.path.exists(EPOCH_TRACKER_PATH):
        with open(EPOCH_TRACKER_PATH, "r") as f:
            return json.load(f).get("epochs_trained", 0)
    return 0

# Save model and training progress
def save_checkpoint(epochs_trained):
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(EPOCH_TRACKER_PATH, "w") as f:
        json.dump({"epochs_trained": epochs_trained}, f)
    print(f"Model and training progress saved (Epochs: {epochs_trained}).")


# Training und Testing
def train_target_model(max_epochs=150, patience=20):  
    current_epochs = load_checkpoint()
    if current_epochs >= max_epochs:
        print(f"Model already trained for {current_epochs} epochs. No further training neccessary.")
        return

    print(f"Resuming training from epoch{current_epochs}).")
    train_losses = []
    train_accuracies = []  
    test_accuracies = []

    best_train_accuracy = 0  # Save best Train Accuracy
    no_improve_epochs = 0

    for epoch in range(current_epochs, max_epochs):
        running_loss = 0.0
        correct_train = 0  # counts correct
        total_train = 0    
        model.train()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # calculate train accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # count avg train and loss
        train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # evaluate train and test data
        test_accuracy = test_target_model()
        test_accuracies.append(test_accuracy)

        print(f"[Epoch {epoch + 1}] Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # Save model only when train accuracy increases
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            save_checkpoint(epoch + 1)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()
        print(f"Learning Rate after Epoch: {epoch + 1}: {scheduler.get_last_lr()}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_PATH)
    plt.close()


def test_target_model():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

#
if __name__ == "__main__":
    train_target_model()
