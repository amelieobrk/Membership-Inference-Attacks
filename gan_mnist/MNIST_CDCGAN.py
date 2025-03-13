#Train a cGan on the MNIST dataset

import os
import time
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#define generator
class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)  # Input: Noise vector (100x1x1)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4 + 10, d * 2, 4, 2, 1)   # Label conditioning (10 channels for class info)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)  
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)   # Output: 32x32 grayscale image

    def forward(self, input, labels):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))  # Expand noise input
        labels = labels.expand(-1, -1, x.size(2), x.size(3))  # Expand labels to match feature size
        x = torch.cat([x, labels], dim=1)  # Concatenate labels with latent features
        x = F.relu(self.deconv2_bn(self.deconv2(x)))  
        x = F.relu(self.deconv3_bn(self.deconv3(x))) 
        x = torch.tanh(self.deconv4(x))  # Normalize output between -1 and 1
        return x

#define discriminator
class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1 + 10, d, 4, 2, 1)  # Input: 32x32 MNIST image + class label
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)   
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)  
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)   # Output: probability score (real or fake)

    def forward(self, input, labels):
        labels = labels.expand(-1, -1, input.size(2), input.size(3)) # Expand labels to match feature maps
        x = torch.cat([input, labels], dim=1)   # Concatenate labels with input image
        x = F.leaky_relu(self.conv1(x), 0.2)
      
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
      
        x = F.leaky_relu(self.conv3(x), 0.2)
       
        x = torch.sigmoid(self.conv4(x))
        print(f"Nach conv4: {x.size()}") # Probability output
        return x.view(-1, 1)  #Reshape for binary classification

# Training Setup
base_dir = os.path.expanduser("~/amelie/gan_mnist")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

results_dir = os.path.join(base_dir, "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

model_dir = os.path.join(base_dir, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = DataLoader(
    datasets.MNIST(base_dir, train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)

# Initialize Models
G = Generator().cuda()
D = Discriminator().cuda()

# Load pre-trained models if available
generator_path = os.path.join(model_dir, "generator.pth")
discriminator_path = os.path.join(model_dir, "discriminator.pth")
epoch_path = os.path.join(model_dir, "last_epoch.txt")
start_epoch = 1
if os.path.exists(generator_path) and os.path.exists(discriminator_path):
    print("Load pre-trained models...")
    G.load_state_dict(torch.load(generator_path))
    D.load_state_dict(torch.load(discriminator_path))
    if os.path.exists(epoch_path):
        with open(epoch_path, "r") as f:
            start_epoch = int(f.read().strip()) + 1
else:
    print("No pre-trained models found. Start new Training...")


# define loss function (BCELoss) and adam optimizer
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
num_epochs = 100 #sufficient for mnist
fixed_noise = torch.randn(100, 100, 1, 1).cuda()
fixed_labels = torch.eye(10).repeat(10, 1).view(100, 10, 1, 1).cuda()

for epoch in range(start_epoch, start_epoch + num_epochs):
    print(f"started epoch {epoch} ...")
    for batch_idx, (real_images, labels) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.cuda()
        real_labels = torch.eye(10)[labels].view(batch_size, 10, 1, 1).cuda()

        # Train Discriminator
        D.zero_grad()
        real_validity = D(real_images, real_labels)
        real_loss = criterion(real_validity, torch.ones(batch_size, 1).cuda())

        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake_labels = torch.eye(10)[torch.randint(0, 10, (batch_size,))].view(batch_size, 10, 1, 1).cuda()
        fake_images = G(noise, fake_labels)
        fake_validity = D(fake_images.detach(), fake_labels)
        fake_loss = criterion(fake_validity, torch.zeros(batch_size, 1).cuda())

        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optimizer.step()

        # Train Generator
        G.zero_grad()
        fake_validity = D(fake_images, fake_labels)
        G_loss = criterion(fake_validity, torch.ones(batch_size, 1).cuda())
        G_loss.backward()
        G_optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx + 1}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")
    
    # Save images at the end of each epoch
    G.eval()
    with torch.no_grad():
        generated_images = G(fixed_noise, fixed_labels)

    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(generated_images[i * 10 + j].squeeze().cpu().numpy(), cmap="gray")
            axs[i, j].axis("off")
            axs[i, j].set_title(f"Label: {j}", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"epoch_{epoch}.png"))
    plt.close()
    G.train()

    # Save the current epoch number
    with open(epoch_path, "w") as f:
        f.write(str(epoch))

# Save the trained model
torch.save(G.state_dict(), os.path.join(model_dir, "generator.pth"))
torch.save(D.state_dict(), os.path.join(model_dir, "discriminator.pth"))

print("Training done and models safed!")
