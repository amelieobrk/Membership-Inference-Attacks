#Load trained Generator model to generate synthetic CIFAR-10 images and organizes them into separate directories based on their class labels.
#This is done to calculate the fid score of different generated classes

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generator-Class
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
        x = torch.relu(self.deconv1_bn(self.deconv1(input)))
        labels = labels.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        x = torch.relu(self.deconv2_bn(self.deconv2(x)))
        x = torch.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x

# CIFAR-10 Classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to generate and organize synthetic images
def generate_and_organize_generated_images(generator, save_dir, num_images_per_class=5000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator.eval()
    with torch.no_grad():
        for class_idx, class_name in enumerate(cifar10_classes):
            class_dir = os.path.join(save_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            for i in range(num_images_per_class):
                noise = torch.randn(1, 100, 1, 1).cuda()
                label = torch.eye(10)[class_idx].view(1, 10, 1, 1).cuda()
                generated_image = generator(noise, label)

                img = (generated_image[0].permute(1, 2, 0).cpu().numpy() + 1) / 2  # Rescale to [0, 1]
                file_name = os.path.join(class_dir, f"generated_{i}.png")
                plt.imsave(file_name, img)

            print(f"{num_images_per_class} Images for class '{class_name}' generated and organized.")

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/amelie/cgan_cifar10/fid_score")
    organized_generated_images_dir = os.path.join(base_dir, "generated_images")

    # Organize generated images
    print("Organize generated images..")
    generator_path = os.path.join(base_dir, "models/generator.pth")
    G = Generator().cuda()
    G.load_state_dict(torch.load(generator_path))

    generate_and_organize_generated_images(G, organized_generated_images_dir, num_images_per_class=5000)
