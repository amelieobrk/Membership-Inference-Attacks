import os
import time
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Fortschrittsanzeige
import numpy as np

# Generator G(z, label)
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

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels).view(-1, 100, 1, 1)
        x = noise * label_embedding
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x

# Discriminator D(x, label)
class Discriminator(nn.Module):
    def __init__(self, d=128, num_classes=2):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 64 * 64)
        self.conv1 = nn.Conv2d(4, d, 4, 2, 1)   # Entferne Label-Kanal
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels).view(-1, 1, 64, 64)
        x = torch.cat((img, label_embedding), 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x

# Training Setup
batch_size = 128
lr = 0.0002
epochs = 50
img_size = 64
data_dir = '~/amelie/data/preprocessed_celebA'
results_dir = '/home/lab24inference/amelie/gan_celeba/epoch_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


dataset = datasets.ImageFolder(data_dir, transform)
for i, (img_path, _) in enumerate(dataset.imgs):
    img = plt.imread(img_path)
    if img.shape[:2] != (img_size, img_size):
        print(f"⚠️ WARNUNG: Bild {img_path} ist nicht {img_size}x{img_size}, sondern {img.shape[:2]}")
    if i >= 5:  # Nach 5 Bildern abbrechen
        break


train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

# Model Init
G = Generator(128).cuda()
D = Discriminator(128).cuda()
BCE_loss = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop

for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)  # Fortschrittsbalken über alle Batches
    for i, (imgs, labels) in enumerate(loop):
        imgs, labels = imgs.cuda(), labels.cuda()
        
        y_real = torch.ones(imgs.size(0), 1).cuda()
        y_fake = torch.zeros(imgs.size(0), 1).cuda()
        
        # Train Discriminator
        D.zero_grad()
        D_real_loss = BCE_loss(D(imgs, labels).view(-1, 1), y_real)
        noise = torch.randn(imgs.size(0), 100, 1, 1).cuda()
        fake_labels = torch.randint(0, 2, (imgs.size(0),)).cuda()
        fake_imgs = G(noise, fake_labels)
        D_fake_loss = BCE_loss(D(fake_imgs.detach(), fake_labels).view(-1, 1), y_fake)
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
        
        # Train Generator
        G.zero_grad()
        G_loss = BCE_loss(D(fake_imgs, fake_labels).view(-1, 1), y_real)
        G_loss.backward()
        G_optimizer.step()

        # Fortschrittsbalken aktualisieren
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    # Speichern der generierten Bilder nach jeder Epoche

    with torch.no_grad():
        n_images = 25  # Anzahl der gewünschten Bilder (5x5 Grid)

        test_noise = torch.randn(n_images, 100, 1, 1).cuda()
        test_labels = torch.cat((torch.zeros(n_images // 2), torch.ones(n_images // 2))).long().cuda()

        if test_labels.shape[0] < n_images:  # Falls n_images ungerade ist
            test_labels = torch.cat((test_labels, torch.randint(0, 2, (1,)).long().cuda()))

        generated_imgs = G(test_noise, test_labels)

        # Setze ein 5x5 Grid mit leichtem Abstand
        grid_size = int(n_images ** 0.5)  
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

        for i, ax in enumerate(axes.flat):
            img = generated_imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # Rescale auf [0,1]

            ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")  # Achsen ausblenden

        fig.subplots_adjust(wspace=0.2, hspace=0.2)  # Leichter Abstand zwischen den Bildern
        plt.savefig(f'{results_dir}/epoch_{epoch+1}.png', bbox_inches="tight", pad_inches=0.1)  # Leichtes Padding für Rahmen
        plt.close()


    print(f'📢 Epoch {epoch+1}/{epochs} | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}')

torch.save(G.state_dict(), "/home/lab24inference/amelie/gan_celeba/generator.pth")
torch.save(D.state_dict(), "/home/lab24inference/amelie/gan_celeba/discriminator.pth")

# bohne