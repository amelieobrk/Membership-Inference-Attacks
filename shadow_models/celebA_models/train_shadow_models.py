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

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
base_dir = os.path.expanduser("~/amelie/shadow_models/celebA_models")
models_dir = os.path.join(base_dir, "models")
progress_dir = os.path.join(base_dir, "progress")  # Store training progress
os.makedirs(models_dir, exist_ok=True)
os.makedirs(progress_dir, exist_ok=True)

# Data Directory
data_dir = os.path.expanduser("~/amelie/shadow_models_data/celebA")

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

# Custom Dataset Class
class CelebANPZDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['images']
        self.labels = data['labels'].astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, np.ndarray) and image.shape[-1] == 3:
            image = Image.fromarray(image.astype('uint8'))

        if self.transform and isinstance(image, Image.Image):
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# Load training and validation datasets
def load_shadow_model_data(model_id):
    train_path = os.path.join(data_dir, f"shadow_model_{model_id}/train/train.npz")
    test_path = os.path.join(data_dir, f"shadow_model_{model_id}/test/test.npz")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Shadow model data not found for ID {model_id}")

    train_dataset = CelebANPZDataset(train_path, transform=transform)
    val_dataset = CelebANPZDataset(test_path, transform=transform)
    return train_dataset, val_dataset

# Define Model
def create_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1)
    )
    return model.to(device)

# Load saved progress and checkpoint
def load_checkpoint(model, optimizer, model_id):
    progress_path = os.path.join(progress_dir, f"progress_shadow_model_{model_id}.json")
    model_path = os.path.join(models_dir, f"shadow_model_{model_id}.pth")

    start_epoch = 0  # Default start

    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
        start_epoch = progress.get("last_epoch", 0) + 1  # Start from next epoch

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Resuming model {model_id} from epoch {start_epoch}.")

    return start_epoch

# Save training progress
def save_progress(model_id, epoch):
    progress_path = os.path.join(progress_dir, f"progress_shadow_model_{model_id}.json")
    progress_data = {"last_epoch": epoch}
    with open(progress_path, "w") as f:
        json.dump(progress_data, f)

# Training function
def train_shadow_model(model, train_loader, val_loader, num_epochs=5, model_id=0):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    start_epoch = load_checkpoint(model, optimizer, model_id)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        # Save progress and model
        save_progress(model_id, epoch)
        model_path = os.path.join(models_dir, f"shadow_model_{model_id}.pth")
        torch.save(model.state_dict(), model_path)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        print(f"Model {model_id} saved at epoch {epoch+1}")

# Main execution
if __name__ == "__main__":
    num_shadow_models = 5  

    for model_id in range(num_shadow_models):
        print(f"\nTraining Shadow Model {model_id}")

        try:
            train_dataset, val_dataset = load_shadow_model_data(model_id)
        except FileNotFoundError as e:
            print(f"Skipping model {model_id}: {e}")
            continue

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        model = create_model()
        train_shadow_model(model, train_loader, val_loader, num_epochs=5, model_id=model_id)

    print("Training complete.")
