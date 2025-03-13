#Generate synthetic MNIST-like images using the trained DCGAN and Split images into shadow model datasets
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
output_dir = os.path.expanduser("~/amelie/shadow_models_data/fake_mnist")
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.expanduser("~/amelie/gan_mnist/models")

# Define Generator architecture
class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4 + 10, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def forward(self, input, labels):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        labels = labels.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x

# Initialize and load the trained Generator model
G = Generator().to(device)

generator_path = os.path.join(model_dir, "generator.pth")
G.load_state_dict(torch.load(generator_path, map_location=device))
G.eval()

# Generate synthetic MNIST-like images and split into shadow model datasets
def generate_fake_mnist_data(total_samples, output_dir, num_splits=20):
    batch_size = 128
    num_batches = total_samples // batch_size
    samples_per_split = total_samples // num_splits  #each shadow model gets equal amount of data
    train_size = samples_per_split // 2  # 50% train, 50% test

    print(f"Starting generation of {total_samples} synthetic mnist images...")

    images = []
    labels = []

    for batch_idx in range(num_batches):
        # Generate random noise and labels for data diversity
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        label_indices = torch.randint(0, 10, (batch_size,), device=device)
        labels_onehot = torch.eye(10, device=device)[label_indices].view(batch_size, 10, 1, 1)

        with torch.no_grad():
            generated_images = G(noise, labels_onehot).cpu().numpy()

        labels_np = label_indices.cpu().numpy()

        images.append(generated_images)
        labels.append(labels_np)

        print(f"Batch {batch_idx + 1}/{num_batches} generated.")

    #Concanate all batches
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    # split data into disjoint subsets for shadow models
    for i in range(num_splits):
        start_idx = i * samples_per_split
        end_idx = start_idx + samples_per_split

        split_images = images[start_idx:end_idx]
        split_labels = labels[start_idx:end_idx]

        # Shuffle before Train/Test-Split
        indices = np.random.permutation(len(split_images))
        split_images = split_images[indices]
        split_labels = split_labels[indices]

        #Divide into training and test sets
        train_images, test_images = split_images[:train_size], split_images[train_size:]
        train_labels, test_labels = split_labels[:train_size], split_labels[train_size:]

        # create data directories for each shadow model
        shadow_model_dir = os.path.join(output_dir, f"shadow_model_{i}")
        train_dir = os.path.join(shadow_model_dir, "train")
        test_dir = os.path.join(shadow_model_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Save in compressed MNIST-like format
        np.savez_compressed(os.path.join(train_dir, "train.npz"), images=train_images, labels=train_labels)
        np.savez_compressed(os.path.join(test_dir, "test.npz"), images=test_images, labels=test_labels)

        print(f"Shadow Model {i+1}/{num_splits} saved: Train={train_images.shape[0]}, Test={test_images.shape[0]}")

    print(f"Successfully generated and saved {total_samples} images!")

# generate 400.000 images
generate_fake_mnist_data(400000, output_dir)
