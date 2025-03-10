import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    def __init__(self, depth=1, regularization=None, reg_constant=0.01):
        super(MNISTConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dropout = nn.Dropout(0.25 if regularization == 'dropout' else 0.0)
        # Update the number of input features to 64 * 6 * 6 = 2304
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data(path):
    data = np.load(path)
    # Ursprünglich war hier ein Fehler, der eine zusätzliche Dimension hinzugefügt hat
    images = torch.tensor(data['images']).float()  # Entferne `.unsqueeze(1)` wenn es nicht benötigt wird
    labels = torch.tensor(data['labels']).long()
    # Stelle sicher, dass die Bilder die richtige Dimension haben
    if images.dim() == 3:  # [N, H, W]
        images = images.unsqueeze(1)  # Füge die Kanaldimension hinzu [N, C, H, W]
    return images, labels


import os

def train_model(model, train_loader, test_loader, device, model_id, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    models_dir = '/home/lab24inference/amelie/shadow_models/mnist_models/models'
    os.makedirs(models_dir, exist_ok=True)  # Stellt sicher, dass das Verzeichnis existiert
    progress_path = os.path.join(models_dir, f'progress_model_{model_id}.json')

    # Load progress if exists
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        start_epoch = progress['epochs']
    else:
        progress = {'epochs': 0, 'train_accuracy': [], 'test_accuracy': []}
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        train_acc = 100 * correct / total
        progress['train_accuracy'].append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        test_acc = 100 * correct / total
        progress['test_accuracy'].append(test_acc)
        progress['epochs'] = epoch + 1
        # Save progress
        with open(progress_path, 'w') as f:
            json.dump(progress, f)
        print(f'Epoch {epoch+1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

     # Save the trained model
        model_save_path = os.path.join(models_dir, f'model_{model_id}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model {model_id} saved at {model_save_path}')


    return progress['train_accuracy'], progress['test_accuracy']


def plot_accuracy(train_acc, test_acc, model_id):
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.title(f'Accuracy Plot for Model {model_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'/home/lab24inference/amelie/shadow_models/mnist_models/models/accuracy_model_{model_id}.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on device: {device}')
    num_models = 20
    for i in range(num_models):
        train_data_path = f'/home/lab24inference/amelie/shadow_models_data/fake_mnist/shadow_model_{i}/train/train.npz'
        test_data_path = f'/home/lab24inference/amelie/shadow_models_data/fake_mnist/shadow_model_{i}/test/test.npz'
        train_images, train_labels = load_data(train_data_path)
        test_images, test_labels = load_data(test_data_path)
        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)
        model = MNISTConvNet(regularization='dropout').to(device)
        train_acc, test_acc = train_model(model, train_loader, test_loader, device, i)
        plot_accuracy(train_acc, test_acc, i)

if __name__ == '__main__':
    main()
