import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# CIFAR-10 Klassenbezeichnungen
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Funktion zur Organisation von realen Bildern
def organize_real_images(real_images_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Transform für die Daten
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # CIFAR-10 Dataset laden
    cifar10_dataset = datasets.CIFAR10(real_images_dir, train=True, download=True, transform=transform)
    data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=False)

    # Bilder nach Klassen organisieren
    for class_name in cifar10_classes:
        class_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    print("Organisiere reale Bilder...")

    for idx, (image, label) in enumerate(data_loader):
        class_name = cifar10_classes[label.item()]
        class_dir = os.path.join(save_dir, class_name)
        file_name = os.path.join(class_dir, f"real_{idx}.png")

        # Bild speichern
        image = image.squeeze().permute(1, 2, 0).numpy()  # Konvertiere Tensor zu NumPy-Array
        plt.imsave(file_name, image)

        if idx % 1000 == 0:
            print(f"{idx} Bilder organisiert...")

    print("Alle realen Bilder erfolgreich organisiert.")

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/amelie/cgan_cifar10")
    real_images_dir = os.path.join(base_dir, "cifar-10-batches-py")  # Pfad zu CIFAR-10-Daten
    organized_real_images_dir = os.path.join(base_dir, "real_images")

    organize_real_images(real_images_dir, organized_real_images_dir)
