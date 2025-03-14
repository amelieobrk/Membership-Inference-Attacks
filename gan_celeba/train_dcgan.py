# training of a dcgan for celebA Pictures

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Fortschrittsanzeige
import numpy as np
import json


model_save_path = "/home/lab24inference/amelie/gan_celeba/models"
state_file = f"{model_save_path}/training_state.json"


if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

#generator class
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
        self.dropout = nn.Dropout(0.2)# dropout -> Don't make generator too dominant


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


class Discriminator(nn.Module):
    def __init__(self, d=128, num_classes=2):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 64 * 64)
        self.conv1 = nn.Conv2d(4, d, 4, 2, 1)   
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)
        self.ln1 = nn.LayerNorm([d, 32, 32])  
        self.ln2 = nn.LayerNorm([d * 2, 16, 16])  
        self.ln3 = nn.LayerNorm([d * 4, 8, 8])  
        self.ln4 = nn.LayerNorm([d * 8, 4, 4])
        self.dropout = nn.Dropout(0.2)# dropout -> Don't make discriminator too dominant

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels).view(-1, 1, 64, 64)
        x = torch.cat((img, label_embedding), 1)
        x = F.leaky_relu(self.ln1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.ln2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.ln3(self.conv3(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.ln4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x


batch_size = 256 # More stable against model collapse than 128 or 512 
epochs = 100
img_size = 64

data_dir = '~/amelie/data/preprocessed_celebA'
results_dir = '/home/lab24inference/amelie/gan_celeba/epoch_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# data preprocessing
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(data_dir, transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

# install models
G = Generator(128).cuda()
D = Discriminator(128).cuda()
BCE_loss = nn.BCELoss()
#higher lr for generator since discriminator tends to be more dominant
G_optimizer = optim.Adam(G.parameters(), lr=0.0003, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-3)


start_epoch = 0
G_losses, D_losses = [], []
if os.path.exists(state_file):
    with open(state_file, "r") as f:
        training_state = json.load(f)
        start_epoch = training_state["epoch"]
        G_losses = training_state["G_losses"]
        D_losses = training_state["D_losses"]
        
    G.load_state_dict(torch.load(f"{model_save_path}/generator.pth"))
    D.load_state_dict(torch.load(f"{model_save_path}/discriminator.pth"))
    print(f" Continue training after epoch {start_epoch}!")

# restart training from last epoch
for epoch in range(start_epoch, epochs):


    loop = tqdm(train_loader, leave=True)
    for i, (imgs, labels) in enumerate(loop):
        imgs, labels = imgs.cuda(), labels.cuda()
        y_real = torch.full((imgs.size(0), 1), 0.9).cuda() # apply label smoothing to make discriminator less dominant
        y_fake = torch.full((imgs.size(0), 1), 0.1).cuda()
        
        D.zero_grad()
        D_loss = BCE_loss(D(imgs, labels).view(-1, 1), y_real) + BCE_loss(D(G(torch.randn(imgs.size(0), 100, 1, 1).cuda(), labels).detach(), labels).view(-1, 1), y_fake)
        D_loss.backward()
        D_optimizer.step()

        G.zero_grad()
        G_loss = BCE_loss(D(G(torch.randn(imgs.size(0), 100, 1, 1).cuda(), labels), labels).view(-1, 1), y_real)
        G_loss.backward()
        G_optimizer.step()

    # save model after each epoch 
    torch.save(G.state_dict(), f"{model_save_path}/generator.pth")
    torch.save(D.state_dict(), f"{model_save_path}/discriminator.pth")


    # show trainings progress
    loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
    loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    # show a grid of example images after each epoch 

    with torch.no_grad():
        n_images = 25  #5x5 grid

        test_noise = torch.randn(n_images, 100, 1, 1).cuda()
        test_labels = torch.cat((torch.zeros(n_images // 2), torch.ones(n_images // 2))).long().cuda()

        if test_labels.shape[0] < n_images:  # n_images = uneven 
            test_labels = torch.cat((test_labels, torch.randint(0, 2, (1,)).long().cuda()))

        generated_imgs = G(test_noise, test_labels)

        grid_size = int(n_images ** 0.5)  
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

        for i, ax in enumerate(axes.flat):
            img = generated_imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # rescale on [0,1]

            ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")  # dont show axes

        #padding between pictures
        fig.subplots_adjust(wspace=0.2, hspace=0.2)  
        plt.savefig(f'{results_dir}/epoch_{epoch+1}.png', bbox_inches="tight", pad_inches=0.1)  
        plt.close()


    print(f'📢 Epoch {epoch+1}/{epochs} | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}')


    # Save final model
    state_file = f"{model_save_path}/training_state.json"


    # save train state after each epoch in json file
    training_state = {
        "epoch": epoch + 1,
        "G_losses": G_losses,
        "D_losses": D_losses
    }

    with open(state_file, "w") as f:
        json.dump(training_state, f, indent=4)

    print(f"model and training state saved after epoch {epoch+1} ")

    # safe loss values
    G_losses.append(G_loss.item())
    D_losses.append(D_loss.item())








