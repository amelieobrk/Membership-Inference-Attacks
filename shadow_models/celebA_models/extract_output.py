import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch import nn
from sklearn.model_selection import train_test_split
from torchvision import models

# Directories
BASE_DIR = "/home/lab24inference/amelie/shadow_models/celebA_models"
SHADOW_DATA_DIR = "/home/lab24inference/amelie/shadow_models_data/celebA"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = "/home/lab24inference/amelie/shadow_models/celebA_models/attack_data"
COMBINED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_attack_data.npz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Shadow Model Architecture (ResNet18)
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(model.fc.in_features, 1)  # Binary classification
    )
    return model.to(device)

# Load dataset from npz files
def load_data(path):
    data = np.load(path)
    images = torch.tensor(data['images']).float()
    labels = torch.tensor(data['labels']).float()  # Binary labels (0 = Non-Smiling, 1 = Smiling)

    if images.dim() == 3:  # Ensure correct shape (Batch, C, H, W)
        images = images.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert grayscale to 3 channels

    return images, labels

# Extract confidence scores
def extract_confidence_scores(shadow_id, train_loader, test_loader):
    model = create_model().to(device)
    model_path = os.path.join(MODEL_SAVE_DIR, f"shadow_model_{shadow_id}.pth")

    if not os.path.exists(model_path):
        print(f"Model {shadow_id} not found. Skipping.")
        return None

    model.load_state_dict(torch.load(model_path))
    model.eval()

    def get_confidences(loader, member_label):
        probabilities, labels, members = [], [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()  # Sigmoid for binary classification
                probabilities.append(probs)
                labels.append(targets.numpy())
                members.extend([member_label] * len(targets))

        return np.vstack(probabilities), np.hstack(labels), np.array(members)

    # Get confidence scores for members (train) and non-members (test)
    train_conf, train_labels, train_members = get_confidences(train_loader, member_label=1)
    test_conf, test_labels, test_members = get_confidences(test_loader, member_label=0)

    print(f"Shadow Model {shadow_id} - Train conf shape: {train_conf.shape}, Test conf shape: {test_conf.shape}")

    # Combine results
    confidences = np.concatenate((train_conf, test_conf), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    members = np.concatenate((train_members, test_members), axis=0)

    return confidences, labels, members

# Main function to process all shadow models
def main():
    all_probabilities, all_labels, all_members = [], [], []

    for shadow_id in range(20):  # 20 shadow models
        train_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/train/train.npz")
        test_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/test/test.npz")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Data for Shadow Model {shadow_id} not found. Skipping.")
            continue  # Skip this model if data is missing

        train_images, train_labels = load_data(train_path)
        test_images, test_labels = load_data(test_path)

        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        print(f"Extracting confidence scores for Shadow Model {shadow_id}...")
        result = extract_confidence_scores(shadow_id, train_loader, test_loader)
        if result is not None:
            confidences, labels, members = result
            all_probabilities.append(confidences)
            all_labels.append(labels)
            all_members.append(members)

    if not all_probabilities:
        print("No confidence scores collected. Exiting.")
        return

    # Combine all extracted confidence scores
    all_probabilities = np.vstack(all_probabilities)  # Shape (N, 1)
    all_labels = np.hstack(all_labels)  # Shape (N,)
    all_members = np.hstack(all_members)  # Shape (N,)

    print(f"Final dataset contains {all_probabilities.shape[0]} samples.")
    print(f"First 5 probability vectors:\n{all_probabilities[:5]}")
    print(f"First 5 labels: {all_labels[:5]}")
    print(f"First 5 membership labels: {all_members[:5]}")

    # Split into train/test for the attack model (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        np.hstack((all_probabilities, all_labels.reshape(-1, 1))),  # Confidence Scores + Labels
        all_members,  # Membership Labels (1=Train Member, 0=Test Non-Member)
        test_size=0.3,
        random_state=42,
        stratify=all_members
    )

    print(f"Train Data Shape: {X_train.shape}, Test Data Shape: {X_test.shape}")

    # Save combined attack dataset
    np.savez(
        COMBINED_OUTPUT_FILE,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print(f"Combined attack data saved at {COMBINED_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
