#load 10 example image batches of test dataset and the predicted label from one shadow modelm to show their predictions

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
base_dir = "/home/lab24inference/amelie/shadow_models/cifar_models"
models_dir = os.path.join(base_dir, "models")
plots_dir = os.path.join(base_dir, "example_prediction_images")
data_dir = "/home/lab24inference/amelie/shadow_models_data/fake_cifar/shadow_data"
os.makedirs(plots_dir, exist_ok=True)

# Define cifar-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the cifar-10 shadow model
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
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load CIFAR-10 data from npz files
def load_data(path):
    data = np.load(path)
    images = data['images'].astype(np.float32) / 255.0
    labels = torch.tensor(data['labels']).long()
    return images, labels

# custom dataset class for loading cifar-10 data
class CIFARDataset(Dataset):
    def __init__(self, npz_path):
        self.images, self.labels = load_data(npz_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = self.transform(image)
        return image, label

# load test dataset
test_path = os.path.join(data_dir, "shadow_model_2/test/test_data.npz")
test_dataset = CIFARDataset(test_path)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4)

# load trained shadow model
model_path = os.path.join(models_dir, "shadow_model_2.pth")
model = ShadowModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# save example batches
def save_example_batches(model, loader, num_batches=10):
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
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            img = (img * 0.5) + 0.5  # Denormalize

            true_label = class_labels[labels[i].item()]
            pred_label = class_labels[preds[i].item()]
            
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.axis("off")

        # Save plot
        batch_save_path = os.path.join(plots_dir, f"batch_{batch_idx+1}.png")
        plt.savefig(batch_save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f" batch {batch_idx+1} saved at: {batch_save_path}")

# Run the function
save_example_batches(model, test_loader, num_batches=10)
print("10 example batches successfully saved!")
