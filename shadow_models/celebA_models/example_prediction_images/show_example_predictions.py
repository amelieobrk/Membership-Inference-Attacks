#load 10 example image batches of test dataset and the predicted label from one shadow modelm to show their predictions
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
base_dir = os.path.expanduser("~/amelie/shadow_models/celebA_models")
models_dir = os.path.join(base_dir, "models")
plots_dir = os.path.join(base_dir, "example_batches")
data_dir = os.path.expanduser("~/amelie/shadow_models_data/celebA")

os.makedirs(plots_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset for .npz Data
class CelebANPZDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['images']  
        self.labels = data['labels']  
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  
        label = self.labels[idx]

        if isinstance(image, np.ndarray):  
            if image.shape[-1] == 3:  # if not yet tensor
                image = Image.fromarray(image.astype('uint8'))  # convert to pil
            else:
                image = torch.tensor(image).float()  # if similar to tensor convert to float

        if self.transform and isinstance(image, Image.Image):  # Transformation only for PIL-Images
            image = self.transform(image)

        return image, torch.tensor(label).float()


# load predefined shadow model
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.1),  
        torch.nn.Linear(model.fc.in_features, 1)
    )
    return model.to(device)

# load test data
test_path = os.path.join(data_dir, "shadow_model_2/test/test.npz")
test_dataset = CelebANPZDataset(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4)

#load model
model_path = os.path.join(models_dir, "shadow_model_18.pth")
model = create_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#generat example batches
def save_example_batches(model, loader, num_batches=1):
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)  # Binary classification

        # Plots erstellen
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            if i >= len(images):
                break
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert in HWC Format
            img = (img + 1) / 2  # Rescale on [0,1]

            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"True: {int(labels[i].item())} | Pred: {preds[i]}")
            ax.axis("off")

        # Speichern
        batch_save_path = os.path.join(plots_dir, f"batch_{batch_idx+1}.png")
        plt.savefig(batch_save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Batch {batch_idx+1} saved at: {batch_save_path}")

# Speichere die Beispiel-Batches
save_example_batches(model, test_loader, num_batches=10)

print(" Successfully saved 10 example images!")
