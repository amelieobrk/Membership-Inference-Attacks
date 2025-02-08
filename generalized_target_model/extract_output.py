### lade das trainierte modell
## lade und normalisiere die test und trainingsdate
### extrahiere wahrscheinlichkeitsvektoren und labels + mitglieder

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

# Pfade
TARGET_MODEL_PATH = "/home/lab24inference/amelie/target_model/target_model.pth"
OUTPUT_FILE = "/home/lab24inference/amelie/target_model/target_attack_data.npz"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Daten vorbereiten
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root="/home/lab24inference/amelie/target_model/data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=512, shuffle=False, num_workers=2)
    
    testset = datasets.CIFAR10(root="/home/lab24inference/amelie/target_model/data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    
    return trainloader, testloader

trainloader, testloader = load_data()

# Zielmodell definieren
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
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Wahrscheinlichkeiten und Labels extrahieren
def extract_probabilities_and_labels():
    # Modell initialisieren und laden
    model = TargetModel().to(device)
    model.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []
    all_members = []

    # Trainingsdaten verarbeiten (Mitglieder)
    print("Verarbeite Trainingsdaten (Mitglieder)...")
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probabilities)
            all_labels.append(labels.numpy())
            all_members.append(np.ones_like(labels.numpy()))  # member=1

    # Testdaten verarbeiten (Nicht-Mitglieder)
    print("Verarbeite Testdaten (Nicht-Mitglieder)...")
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probabilities)
            all_labels.append(labels.numpy())
            all_members.append(np.zeros_like(labels.numpy()))  # member=0

    # Kombiniere alle Ergebnisse
    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)
    all_members = np.hstack(all_members)

    # Speichere die Ergebnisse
    np.savez(OUTPUT_FILE, probabilities=all_probs, labels=all_labels, members=all_members)
    print(f"Wahrscheinlichkeiten, Labels und Mitgliedschaftskennzeichnung wurden gespeichert unter: {OUTPUT_FILE}")

# Funktion ausführen
if __name__ == "__main__":
    extract_probabilities_and_labels()
