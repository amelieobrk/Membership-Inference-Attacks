"""
Extract probability vectors and membership labels from a trained target model on CIFAR-10.
This script loads a trained target model, processes the CIFAR-10 dataset, and extracts 
probability vectors along with their corresponding labels. It also labels whether each
example was part of the training set (member) or not (non-member) for membership inference attacks.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

# Paths to model and output files
TARGET_MODEL_PATH = "/home/lab24inference/amelie/target_model/target_model.pth"
OUTPUT_FILE = "/home/lab24inference/amelie/target_model/target_attack_data.npz"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-10 dataset
# Training and test sets are processed separately to distinguish members from non-members
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load training set (members)
    trainset = datasets.CIFAR10(root="/home/lab24inference/amelie/target_model/data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=512, shuffle=False, num_workers=2)
    
    # Load test set (non-members)
    testset = datasets.CIFAR10(root="/home/lab24inference/amelie/target_model/data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    
    return trainloader, testloader

trainloader, testloader = load_data()

# Define the same model architecture as the target model
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
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)  # Output logits
        return x

# Extract probability vectors and membership labels
def extract_probabilities_and_labels():
    # Initialize model and load trained parameters
    model = TargetModel().to(device)
    model.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=device))
    model.eval()

    all_probs = []  # Store probability vectors
    all_labels = []  # Store ground truth labels
    all_members = []  # Store membership indicators (1 for train, 0 for test)

    # Process training data (Members)
    print("Processing training data (Members)...")
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probabilities)
            all_labels.append(labels.numpy())
            all_members.append(np.ones_like(labels.numpy()))  # Mark as members (1)

    # Process test data (Non-members)
    print("Processing test data (Non-Members)...")
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probabilities)
            all_labels.append(labels.numpy())
            all_members.append(np.zeros_like(labels.numpy()))  # Mark as non-members (0)

    # Stack all collected data into arrays
    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)
    all_members = np.hstack(all_members)

    # Save the extracted information to a .npz file for later use
    np.savez(OUTPUT_FILE, probabilities=all_probs, labels=all_labels, members=all_members)
    print(f"Probability vectors, labels, and membership indicators saved at: {OUTPUT_FILE}")

# Execute the extraction process
if __name__ == "__main__":
    extract_probabilities_and_labels()
