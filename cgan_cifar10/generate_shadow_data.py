#load a trained generator model, generate 400,000 synthetic CIFAR-10 images,
# and saves them in batches for shadow model training


import os
import torch
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4 + 10, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    def forward(self, input, labels):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        labels = labels.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x

# Setup
output_dir = os.path.expanduser("~/amelie/shadow_models_data/fake_cifar/shadow_data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_dir = os.path.expanduser("~/amelie/cgan_cifar10/models")
generator_path = os.path.join(model_dir, "generator.pth")


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize Generator
G = Generator().cuda()
if os.path.exists(generator_path):
    print("Loading trained Generator-Model...")
    G.load_state_dict(torch.load(generator_path))
else:
    raise FileNotFoundError(f"Couldn't find pre-trained model at:{generator_path}")

#Set generator to evaluation mode
G.eval()


num_images = 400000
batch_size = 128
images_per_file = 50000 

# Geneerate images and safe them as batches
print("generating images for shadow models...")
with torch.no_grad():
    for batch_index in range(num_images // images_per_file):
        batch_images = []
        batch_labels = []

        for _ in range(images_per_file // batch_size):
            noise = torch.randn(batch_size, 100, 1, 1).cuda()
            labels = torch.randint(0, 10, (batch_size,)).cuda()
            one_hot_labels = torch.eye(10, device=labels.device)[labels].view(batch_size, 10, 1, 1)
            generated_images = G(noise, one_hot_labels)

            batch_images.append(generated_images.cpu())
            batch_labels.extend(labels.cpu().numpy())

        # convert images and labels into numpy array
        batch_images = torch.cat(batch_images, dim=0).permute(0, 2, 3, 1).numpy()  # [N, H, W, C]
        batch_images = ((batch_images + 1) / 2 * 255).astype(np.uint8)  # Scale to [0, 255]
        batch_labels = np.array(batch_labels)

        # Save batch files as .npz
        batch_file = os.path.join(output_dir, f"data_batch_{batch_index + 1}.npz")
        np.savez(batch_file, images=batch_images, labels=batch_labels)
        print(f"Batch {batch_index + 1} gespeichert: {batch_file}")

# Save example image for validation
for i in range(10):
    example_noise = torch.randn(1, 100, 1, 1).cuda()
    example_label_index = torch.randint(0, 10, (1,)).cuda()
    example_one_hot = torch.eye(10, device=example_label_index.device)[example_label_index].view(1, 10, 1, 1)
    example_image = G(example_noise, example_one_hot)

    example_path = os.path.join(output_dir, "example_image_with_label{i}.png")
    example_label = cifar10_classes[example_label_index.item()]

# Plot and save the example image with label
plt.figure(figsize=(3, 3))
plt.imshow((example_image[0].detach().permute(1, 2, 0).cpu().numpy() + 1) / 2)
plt.title(f"Label: {example_label}")
plt.axis("off")
plt.savefig(example_path)
plt.close()

print(f"400.000 images generated and saved at: {output_dir}")
print(f"Example image with label saved at: {example_path}.")
