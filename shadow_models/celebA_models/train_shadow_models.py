import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
base_dir = os.path.expanduser("~/amelie/shadow_models/celebA_models")
models_dir = os.path.join(base_dir, "models")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Data Directory
data_dir = os.path.expanduser("~/amelie/shadow_models_data/celebA")

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset for .npz files
from PIL import Image

class CelebANPZDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['images']  # Shape: (N, H, W, C) oder (N, C, H, W)
        self.labels = data['labels']  # Shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Numpy-Array
        label = self.labels[idx]

        # Überprüfen, ob das Bild die Form (H, W, C) hat (statt (C, H, W))
        if image.shape[-1] == 3:  # Falls letzter Index 3 ist, dann ist es (H, W, C)
            image = Image.fromarray(image.astype('uint8'))  # In PIL-Image umwandeln
        else:
            image = torch.tensor(image)  # Falls schon (C, H, W), direkt als Tensor nutzen

        if self.transform:
            image = self.transform(image)  # Anwenden der Transformationen

        return image, torch.tensor(label)


        

# Load train/test datasets for a given shadow model
def load_shadow_model_data(model_id):
    train_path = os.path.join(data_dir, f"shadow_model_{model_id}/train/train.npz")
    test_path = os.path.join(data_dir, f"shadow_model_{model_id}/test/test.npz")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Shadow model data not found for ID {model_id} in {data_dir}")

    train_dataset = CelebANPZDataset(train_path, transform=transform)
    val_dataset = CelebANPZDataset(test_path, transform=transform)

    return train_dataset, val_dataset

# Define Shadow Model Architecture
def create_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.1),  
        nn.Linear(model.fc.in_features, 1)  # Binary classification
    )
    return model.to(device)

# Define loss function and optimizer
def train_shadow_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, model_id=0):
    best_val_loss = float("inf")
    patience = 15  # Early stopping patience
    patience_counter = 0

    progress_path = os.path.join(models_dir, f"progress_shadow_model_{model_id}.json")
    model_save_path = os.path.join(models_dir, f"shadow_model_{model_id}.pth")

    # Load previous training progress if exists
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
        start_epoch = progress['epochs']
        train_loss_history = progress['train_loss']
        val_loss_history = progress['val_loss']
        train_accuracy_history = progress['train_accuracy']
        val_accuracy_history = progress['val_accuracy']
        model.load_state_dict(torch.load(model_save_path))
    else:
        progress = {'epochs': 0, 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        start_epoch = 0
        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        progress.update({
            'epochs': epoch + 1,
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'train_accuracy': train_accuracy_history,
            'val_accuracy': val_accuracy_history
        })
        with open(progress_path, "w") as f:
            json.dump(progress, f)

    # Save loss and accuracy plots
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title(f'Shadow Model {model_id} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"loss_shadow_model_{model_id}.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(train_accuracy_history, label='Train Accuracy')
    plt.plot(val_accuracy_history, label='Validation Accuracy')
    plt.title(f'Shadow Model {model_id} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"accuracy_shadow_model_{model_id}.png"))
    plt.close()

if __name__ == "__main__":
    num_shadow_models = 20  

    for model_id in range(num_shadow_models):
        print(f"Starting training for Shadow Model {model_id}")

        try:
            train_dataset, val_dataset = load_shadow_model_data(model_id)
        except FileNotFoundError as e:
            print(f"Skipping Shadow Model {model_id}: {e}")
            continue  

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        model = create_model()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_shadow_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, model_id=model_id)

        print(f"Finished training for Shadow Model {model_id}\n")

    print("Shadow model training complete!")
