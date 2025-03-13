#organize real cifar-10 images into seperate directories based on their classes for fid-score calculation

import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to organize images
def organize_real_images(real_images_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define transformation for dataset loading
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load Cifar-10 dataset
    cifar10_dataset = datasets.CIFAR10(real_images_dir, train=True, download=True, transform=transform)
    data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=False)

    # Create directories for each class
    for class_name in cifar10_classes:
        class_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    print("Organizing real images...")

    for idx, (image, label) in enumerate(data_loader):
        class_name = cifar10_classes[label.item()]
        class_dir = os.path.join(save_dir, class_name)
        file_name = os.path.join(class_dir, f"real_{idx}.png")

        # Save image
        image = image.squeeze().permute(1, 2, 0).numpy()  # Convert tensor to numpy array
        plt.imsave(file_name, image)

        if idx % 1000 == 0:
            print(f"{idx} Images organized...")

    print("All real CIFAR-10 images successfully organized.")

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/amelie/cgan_cifar10")
    real_images_dir = os.path.join(base_dir, "cifar-10-batches-py")  
    organized_real_images_dir = os.path.join(base_dir, "real_images")

    organize_real_images(real_images_dir, organized_real_images_dir)
