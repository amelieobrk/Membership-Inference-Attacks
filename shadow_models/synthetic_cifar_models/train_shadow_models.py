import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Verzeichnis-Konfiguration
BASE_DIR = "/home/lab24inference/amelie/shadow_models/synthetic_cifar_models"
SHADOW_DATA_DIR = "/home/lab24inference/amelie/shadow_models_data/fake_cifar/shadow_data"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Prüfen, ob GPU verfügbar ist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ShadowModel(nn.Module):
    def __init__(self):
        super(ShadowModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh(),
            nn.Linear(128, 10)  # Keine Softmax hier
        )

        '''
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
'''
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

'''
# Shadow Model Definition
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
'''
# Dataset-Klasse
class ShadowDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        data = np.load(data_path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        label = self.labels[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))
        return self.transform(image), label

# Test-Funktion
def test_shadow_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_loss /= len(test_loader)
    return test_loss, accuracy

# Shadow Model Training
def train_shadow_model(shadow_id, train_loader, test_loader, epochs=15):
    model = ShadowModel().to(device)
    model_path = os.path.join(MODEL_SAVE_DIR, f"shadow_model_{shadow_id}.pth")
    plot_path = os.path.join(PLOT_DIR, f"shadow_model_{shadow_id}_metrics.png")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Höhere Lernrate
   ## scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = test_shadow_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        scheduler.step(train_loss)

        print(f"[Shadow Model {shadow_id}] Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    # Ergebnisse plotten
    plt.figure(figsize=(10, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.savefig(plot_path)
    plt.close()

    torch.save(model.state_dict(), model_path)
    print(f"[Shadow Model {shadow_id}] Training abgeschlossen. Ergebnisse gespeichert unter: {plot_path}")

if __name__ == "__main__":
    for shadow_id in range(1, 31):
        train_data_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/train/train_data.npz")
        test_data_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/test/test_data.npz")

        train_dataset = ShadowDataset(train_data_path, is_train=True)
        test_dataset = ShadowDataset(test_data_path, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

        print(f"Training Shadow Model {shadow_id}...")
        train_shadow_model(shadow_id, train_loader, test_loader)
