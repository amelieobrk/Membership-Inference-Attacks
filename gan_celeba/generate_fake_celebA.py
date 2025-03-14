import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
output_dir = os.path.expanduser("~/amelie/shadow_models_data/celebA")
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.expanduser("~/amelie/gan_celeba/models")
results_dir = os.path.expanduser("~/amelie/gan_celeba/results")
os.makedirs(results_dir, exist_ok=True)

# Define Generator class
class Generator(nn.Module):
    def __init__(self, d=128, num_classes=2):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 100)
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        self.dropout = nn.Dropout(0.2)  # Dropout to prevent discriminator dominance

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels).view(-1, 100, 1, 1)
        x = noise * label_embedding
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.dropout(x)
        x = torch.tanh(self.deconv5(x))
        return x

# Load trained Generator
G = Generator().to(device)
generator_path = os.path.join(model_dir, "generator.pth")
G.load_state_dict(torch.load(generator_path, map_location=device))
G.eval()

# Generate synthetic CelebA-like images
def generate_fake_celebA_data(total_samples, output_dir, num_splits=20):
    batch_size = 128
    num_batches = total_samples // batch_size
    samples_per_split = total_samples // num_splits
    train_size = samples_per_split // 2  # 50% train, 50% test

    print(f"Starting generation of {total_samples} synthetic CelebA images...")

    images = []
    labels = []

    for batch_idx in range(num_batches):
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        label_indices = torch.randint(0, 2, (batch_size,), device=device)  # 0: Non-Smiling, 1: Smiling

        with torch.no_grad():
            generated_images = G(noise, label_indices).cpu().numpy()

        labels_np = label_indices.cpu().numpy()

        images.append(generated_images)
        labels.append(labels_np)
        print(f"Batch {batch_idx + 1}/{num_batches} generated.")

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    for i in range(num_splits):
        start_idx = i * samples_per_split
        end_idx = start_idx + samples_per_split
        split_images = images[start_idx:end_idx]
        split_labels = labels[start_idx:end_idx]

        indices = np.random.permutation(len(split_images))
        split_images = split_images[indices]
        split_labels = split_labels[indices]

        train_images, test_images = split_images[:train_size], split_images[train_size:]
        train_labels, test_labels = split_labels[:train_size], split_labels[train_size:]

        shadow_model_dir = os.path.join(output_dir, f"shadow_model_{i}")
        train_dir = os.path.join(shadow_model_dir, "train")
        test_dir = os.path.join(shadow_model_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        np.savez_compressed(os.path.join(train_dir, "train.npz"), images=train_images, labels=train_labels)
        np.savez_compressed(os.path.join(test_dir, "test.npz"), images=test_images, labels=test_labels)

        print(f"Shadow Model {i+1}/{num_splits} saved: Train={train_images.shape[0]}, Test={test_images.shape[0]}")

    print(f"Successfully generated and saved {total_samples} images!")

# Generate 400,000 images
generate_fake_celebA_data(400000, output_dir)

# Display example batches
def show_example_images():
    noise = torch.randn(16, 100, 1, 1, device=device)
    labels = torch.cat((torch.zeros(8), torch.ones(8))).long().to(device)
    with torch.no_grad():
        generated_images = G(noise, labels).cpu().numpy()
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = generated_images[i].transpose(1, 2, 0)
        img = (img + 1) / 2
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title("Smiling" if labels[i].item() == 1 else "Non-Smiling")
        ax.axis("off")
    
    plt.savefig(os.path.join(results_dir, "example_generated_images.png"))
    print(f"Example images saved to {results_dir}/example_generated_images.png")

show_example_images()
