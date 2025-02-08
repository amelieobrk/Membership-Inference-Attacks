import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Pfade
base_dir = os.path.expanduser("~/amelie/gan_mnist")
generated_dir = os.path.join(base_dir, "generated_cgan_data")
os.makedirs(generated_dir, exist_ok=True)
model_dir = os.path.join(base_dir, "models")

# Generator laden
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

# Modell initialisieren und laden
G = Generator().cuda()
generator_path = os.path.join(model_dir, "generator.pth")
G.load_state_dict(torch.load(generator_path))
G.eval()

# Daten generieren und anzeigen
def generate_and_display(num_batches, batch_size=100):
    for batch_idx in range(num_batches):
        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        label_indices = torch.randint(0, 10, (batch_size,))
        labels_onehot = torch.eye(10).to(noise.device)[label_indices].view(batch_size, 10, 1, 1)

        with torch.no_grad():
            generated_images = G(noise, labels_onehot).cpu()

        # Bilder anzeigen
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                idx = i * 10 + j
                axs[i, j].imshow(generated_images[idx, 0], cmap="gray")
                axs[i, j].axis("off")
                axs[i, j].set_title(f"Label: {label_indices[idx].item()}", fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(generated_dir, f"generated_batch_{batch_idx + 1}.png"))
        plt.close()
        print(f"Batch {batch_idx + 1} von 10 gespeichert.")

# Zeige 1000 Daten in 100er Schritten
generate_and_display(num_batches=10, batch_size=100)
