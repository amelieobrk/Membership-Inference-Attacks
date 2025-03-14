#calculate FID score per class between real and generated images
# using an Inception v3 model for feature extraction.

import os
import torch
import json
import numpy as np
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
from PIL import Image

# Function to load the Inception v3 model for feature extraction
def load_inception_model(device):
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  #Remove classification layer
    model.eval()
    model.to(device)
    return model

# Function to extract features using Inception v3
def extract_features(model, image_dir, transform, device, batch_size=32):
    features = []
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [transform(Image.open(img_path).convert('RGB')) for img_path in batch_paths]
            images = torch.stack(images).to(device)
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features

# Function to calculate the FID score
def calculate_fid(mu1, sigma1, mu2, sigma2):
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Main function to compute FID per class
if __name__ == "__main__":
    base_dir = os.path.expanduser("~/amelie/cgan_cifar10/fid_score")
    real_images_dir = os.path.join(base_dir, "real_images")
    generated_images_dir = os.path.join(base_dir, "generated_images")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load inception model
    print("Loading inception model...")
    model = load_inception_model(device)

    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

     # Compute FID scores per class
    class_fid_scores = {}

   # Request epoch information
    epoch = input("please enter the current epoch number: ")

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    for class_name in classes:
        print(f"Calculate FID score for class: {class_name}...")

        real_class_dir = os.path.join(real_images_dir, class_name)
        generated_class_dir = os.path.join(generated_images_dir, class_name)

        if not os.path.exists(real_class_dir) or not os.path.exists(generated_class_dir):
            print(f"WARNING: class {class_name} is missing in one of the directories. Skipping... ")
            continue

        real_features = extract_features(model, real_class_dir, inception_transform, device)
        generated_features = extract_features(model, generated_class_dir, inception_transform, device)

        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

        fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
        class_fid_scores[class_name] = fid_score
        print(f"FID score for class:{class_name}: {fid_score:.4f}")

    # Save results in json file
    results_path = os.path.join(base_dir, "fid_score", "fid_scores.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            fid_data = json.load(f)
    else:
        fid_data = {}

    fid_data[f"epoch_{epoch}"] = class_fid_scores

    with open(results_path, "w") as f:
        json.dump(fid_data, f, indent=4)

    print(f"FID scores per class saved at: {results_path}.")
