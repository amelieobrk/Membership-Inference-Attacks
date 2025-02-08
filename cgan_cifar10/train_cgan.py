import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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

class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3 + 10, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def forward(self, input, labels):
        labels = labels.expand(-1, -1, input.size(2), input.size(3))
        x = torch.cat([input, labels], dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x.view(-1, 1)

# Training Setup
base_dir = os.path.expanduser("~/amelie/cgan_cifar10")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

results_dir = os.path.join(base_dir, "epoch_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

model_dir = os.path.join(base_dir, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader = DataLoader(
    datasets.CIFAR10(base_dir, train=True, download=True, transform=transform),
    batch_size=256, shuffle=True
)

# CIFAR-10 Klassen
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize Models
G = Generator().cuda()
D = Discriminator().cuda()

# Load pre-trained models if available
generator_path = os.path.join(model_dir, "generator.pth")
discriminator_path = os.path.join(model_dir, "discriminator.pth")
epoch_path = os.path.join(model_dir, "last_epoch.txt")
start_epoch = 1
if os.path.exists(generator_path) and os.path.exists(discriminator_path):
    print("Lade vortrainierte Modelle...")
    G.load_state_dict(torch.load(generator_path))
    D.load_state_dict(torch.load(discriminator_path))
    if os.path.exists(epoch_path):
        with open(epoch_path, "r") as f:
            start_epoch = int(f.read().strip()) + 1
else:
    print("Keine vortrainierten Modelle gefunden. Beginne neues Training...")

criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define a function to adjust the learning rate
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training Loop
num_epochs = 300#das hier sind die ZUSÄTZLICHEN EPochen , start e ist die zuletzt gespiecherte!
lr_reduction_epoch1 =  190  # Epoche, nach der die Lernrate reduziert wird
lr_reduction_epoch2 =  240
lr_reduction_epoch3 = 270

fixed_noise = torch.randn(10, 100, 1, 1).cuda()
fixed_labels = torch.eye(10).view(10, 10, 1, 1).cuda()

for epoch in range(start_epoch, start_epoch + num_epochs):
    print(f"Epoch {epoch} gestartet...")

    # Reduziere die Lernrate nach lr_reduction_epoch
    if epoch == lr_reduction_epoch1:
        new_lr = G_optimizer.param_groups[0]['lr'] * 0.75  # reduziere die lernrate etwas
        adjust_learning_rate(D_optimizer, new_lr)
        print(f"Lernrate reduziert auf {new_lr}")
    if epoch == lr_reduction_epoch2:
        new_lr = G_optimizer.param_groups[0]['lr'] * 0.5  # reduziere die lernrate etwas
        adjust_learning_rate(D_optimizer, new_lr)
        print(f"Lernrate reduziert auf {new_lr}")

    if epoch == lr_reduction_epoch3:
        new_lr = G_optimizer.param_groups[0]['lr'] * 0.25  # reduziere die lernrate etwas
        adjust_learning_rate(D_optimizer, new_lr)
        print(f"Lernrate reduziert auf {new_lr}")

    for batch_idx, (real_images, labels) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.cuda()
        real_labels = torch.eye(10)[labels].view(batch_size, 10, 1, 1).cuda()

        # Label Smoothing
        smooth_real = torch.full((batch_size, 1), 0.95).cuda()  # Echte Labels (0.9 statt 1.0)
        smooth_fake = torch.full((batch_size, 1), 0.25).cuda()  # Falsche Labels (0.1 statt 0.0)

        # Train Discriminator
        D.zero_grad()
        real_validity = D(real_images, real_labels)
        real_loss = criterion(real_validity, smooth_real)

        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake_labels = torch.eye(10)[torch.randint(0, 10, (batch_size,))].view(batch_size, 10, 1, 1).cuda()
        fake_images = G(noise, fake_labels)
        fake_validity = D(fake_images.detach(), fake_labels)
        fake_loss = criterion(fake_validity, smooth_fake)

        D_loss = ( real_loss + fake_loss ) * 1.1
        D_loss.backward()
        D_optimizer.step()

        # Train Generator
        G.zero_grad()
        fake_validity = D(fake_images, fake_labels)
        G_loss = criterion(fake_validity, torch.ones(batch_size, 1).cuda())
        G_loss.backward()
        G_optimizer.step()

        # Loggen der Verluste
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx + 1}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")

    # Generiere Bilder zum Visualisieren
    G.eval()
    with torch.no_grad():
        diverse_noise = torch.randn(30, 100, 1, 1).cuda()
        diverse_labels = torch.eye(10)[torch.randint(0, 10, (30,))].view(30, 10, 1, 1).cuda()
        generated_images = G(diverse_noise, diverse_labels)

    fig, axs = plt.subplots(3, 10, figsize=(15, 5))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow((generated_images[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)
        ax.axis("off")
        ax.set_title(f"Label: {cifar10_classes[diverse_labels[i].argmax().item()]}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"epoch_{epoch}.png"))
    plt.close()
    G.train()

    # Speichere Modelle
    torch.save(G.state_dict(), os.path.join(model_dir, "generator.pth"))
    torch.save(D.state_dict(), os.path.join(model_dir, "discriminator.pth"))
    with open(epoch_path, "w") as f:
        f.write(str(epoch))

print("Training abgeschlossen und Modelle gespeichert!")