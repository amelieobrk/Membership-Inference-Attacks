import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Verzeichnis-Konfiguration
BASE_DIR = "/home/lab24inference/amelie/shadow_models/generalized_synthetic_cifar_models"
SHADOW_DATA_DIR = "/home/lab24inference/amelie/shadow_models_data/fake_cifar/shadow_data"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
PLOTS_SAVE_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)
EPOCH_TRACKER_PATH = os.path.join(MODEL_SAVE_DIR, "epoch_tracker.json")

# Prüfen, ob GPU verfügbar ist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Shadow Model Definition
class ShadowModel(nn.Module):
    def __init__(self):
        super(ShadowModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Dataset-Klasse
class ShadowDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        data = np.load(data_path)
        self.images = data["images"]
        self.labels = data["labels"]
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
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

# Lade gespeicherte Epoche und beste Testgenauigkeit
def load_checkpoint(shadow_id):
    default = {"epoch": 0, "best_test_acc": 0}
    if os.path.exists(EPOCH_TRACKER_PATH):
        with open(EPOCH_TRACKER_PATH, "r") as f:
            epoch_data = json.load(f)
        return epoch_data.get(str(shadow_id), default)
    return default

# Speichere trainierte Epoche und beste Testgenauigkeit
def save_checkpoint(shadow_id, epoch, best_test_acc):
    if os.path.exists(EPOCH_TRACKER_PATH):
        with open(EPOCH_TRACKER_PATH, "r") as f:
            epoch_data = json.load(f)
    else:
        epoch_data = {}

    epoch_data[str(shadow_id)] = {
        "epoch": epoch,
        "best_test_acc": best_test_acc
    }

    with open(EPOCH_TRACKER_PATH, "w") as f:
        json.dump(epoch_data, f)

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

# Training mit Speicherung der Epoche
def train_shadow_model(shadow_id, train_loader, test_loader, max_epochs=200):
    model = ShadowModel().to(device)
    model_path = os.path.join(MODEL_SAVE_DIR, f"shadow_model_{shadow_id}.pth")

    # Lade gespeicherte Epoche und beste Testgenauigkeit
    checkpoint_data = load_checkpoint(shadow_id)
    current_epoch = checkpoint_data["epoch"]
    best_test_accuracy = checkpoint_data["best_test_acc"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Listen für das Speichern von Trainings- und Testdaten
    train_losses, test_losses, test_accuracies, train_accuracies = [], [], [], []

    print(f"[Shadow Model {shadow_id}] Fortsetzung des Trainings ab Epoche {current_epoch}.")

    for epoch in range(current_epoch, max_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_loss, test_accuracy = test_shadow_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"[Shadow Model {shadow_id}] Epoch {epoch+1}: Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), model_path)
            save_checkpoint(shadow_id, epoch, best_test_accuracy)
            print(f"[Shadow Model {shadow_id}] Neue beste Testgenauigkeit: {best_test_accuracy:.2f}% bei Epoche {epoch}. Modell gespeichert.")

        scheduler.step()

    # Trainingsergebnisse speichern
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
    plt.savefig(os.path.join(PLOTS_SAVE_DIR, f"shadow_model_{shadow_id}_plot.png"))
    plt.close()

    print(f"[Shadow Model {shadow_id}] Training abgeschlossen. Plot gespeichert unter: {PLOTS_SAVE_DIR}")

# Starte Training für alle Shadow Models
if __name__ == "__main__":
    for shadow_id in range(1, 31):
        train_data_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/train/train_data.npz")
        test_data_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/test/test_data.npz")

        train_dataset = ShadowDataset(train_data_path, is_train=True)
        test_dataset = ShadowDataset(test_data_path, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

        train_shadow_model(shadow_id, train_loader, test_loader)
