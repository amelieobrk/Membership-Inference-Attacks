import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Gerät setzen (GPU bevorzugt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Hauptverzeichnis für Fake MNIST-Daten
output_dir = os.path.expanduser("~/amelie/shadow_models_data/fake_mnist")
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.expanduser("~/amelie/gan_mnist/models")

# Generator-Architektur definieren
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

# Modell initialisieren und auf GPU/CPU laden
G = Generator().to(device)

generator_path = os.path.join(model_dir, "generator.pth")
G.load_state_dict(torch.load(generator_path, map_location=device))
G.eval()

# Fake MNIST Daten generieren
def generate_fake_mnist_data(total_samples, output_dir, num_splits=20):
    batch_size = 128
    num_batches = total_samples // batch_size
    samples_per_split = total_samples // num_splits  # 10.000 pro Shadow Model
    train_size = samples_per_split // 2  # 5.000 Train, 5.000 Test

    print(f"Starte Generierung von {total_samples} Fake-MNIST-Bildern...")

    images = []
    labels = []

    for batch_idx in range(num_batches):
        # Zufallsrauschen und Labels generieren
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        label_indices = torch.randint(0, 10, (batch_size,), device=device)
        labels_onehot = torch.eye(10, device=device)[label_indices].view(batch_size, 10, 1, 1)

        with torch.no_grad():
            generated_images = G(noise, labels_onehot).cpu().numpy()

        labels_np = label_indices.cpu().numpy()

        images.append(generated_images)
        labels.append(labels_np)

        print(f"Batch {batch_idx + 1}/{num_batches} generiert.")

    # Alle Batches zusammenfügen
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Daten in 20 disjunkte Splits aufteilen
    for i in range(num_splits):
        start_idx = i * samples_per_split
        end_idx = start_idx + samples_per_split

        split_images = images[start_idx:end_idx]
        split_labels = labels[start_idx:end_idx]

        # Shuffle vor dem Train/Test-Split
        indices = np.random.permutation(len(split_images))
        split_images = split_images[indices]
        split_labels = split_labels[indices]

        # In 50% Train / Test aufteilen
        train_images, test_images = split_images[:train_size], split_images[train_size:]
        train_labels, test_labels = split_labels[:train_size], split_labels[train_size:]

        # Speicherpfade erstellen
        shadow_model_dir = os.path.join(output_dir, f"shadow_model_{i}")
        train_dir = os.path.join(shadow_model_dir, "train")
        test_dir = os.path.join(shadow_model_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Speichern im MNIST-Format
        np.savez_compressed(os.path.join(train_dir, "train.npz"), images=train_images, labels=train_labels)
        np.savez_compressed(os.path.join(test_dir, "test.npz"), images=test_images, labels=test_labels)

        print(f"✅ Shadow Model {i+1}/{num_splits} gespeichert: Train={train_images.shape[0]}, Test={test_images.shape[0]}")

    print(f"🎉 Alle {total_samples} Bilder wurden erfolgreich gespeichert!")

# Generiere 200.000 Daten in 20 Splits mit Train/Test-Splits und eigener Ordnerstruktur
generate_fake_mnist_data(400000, output_dir)
