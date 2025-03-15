#load 10 example image batches of test dataset and the predicted label from one shadow modelm to show their predictions

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
base_dir = "/home/lab24inference/amelie/shadow_models/mnist_models"
models_dir = os.path.join(base_dir, "models")
plots_dir = os.path.join(base_dir, "example_prediction_images")
data_dir = "/home/lab24inference/amelie/shadow_models_data/fake_mnist"
os.makedirs(plots_dir, exist_ok=True)

# Define the MNIST model
class MNISTConvNet(nn.Module):
    def __init__(self):
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
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST data from .npz files
def load_data(path):
    data = np.load(path)
    images = torch.tensor(data['images']).float()
    labels = torch.tensor(data['labels']).long()
    if images.dim() == 3:
        images = images.unsqueeze(1)  # Add channel dimension
    return images, labels

# Load test dataset
test_path = os.path.join(data_dir, "shadow_model_4/test/test.npz") #should be different than loaded model!
test_images, test_labels = load_data(test_path)
test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4)

# Load the trained shadow model
model_path = os.path.join(models_dir, "model_2.pth")
model = MNISTConvNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Save example batches
def save_example_batches(model, loader, num_batches=1):
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get predicted labels

        # Plot
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i >= len(images):
                break
            img = images[i].cpu().numpy().squeeze()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {labels[i].item()} | Pred: {preds[i].item()}")
            ax.axis("off")

        # Save plot
        batch_save_path = os.path.join(plots_dir, f"batch_{batch_idx+1}.png")
        plt.savefig(batch_save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Batch {batch_idx+1} saved at: {batch_save_path}")

# Run function
save_example_batches(model, test_loader, num_batches=10)
print("10 example images successfully saved")
