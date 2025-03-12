import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch import nn
from sklearn.model_selection import train_test_split

# Directories for model storage and output data
BASE_DIR = "/home/lab24inference/amelie/shadow_models/mnist_models"
SHADOW_DATA_DIR = "/home/lab24inference/amelie/shadow_models_data/fake_mnist"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "attack_data")
COMBINED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_attack_data.npz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset from npz files
def load_data(path):
    data = np.load(path)
    images = torch.tensor(data['images']).float()
    labels = torch.tensor(data['labels']).long()
    if images.dim() == 3:
        images = images.unsqueeze(1)
    return images, labels

# Evaluate model and extract confidence scores
def extract_confidence_scores(shadow_id, train_loader, test_loader):
    model = MNISTConvNet().to(device)
    model_path = os.path.join(MODEL_SAVE_DIR, f"model_{shadow_id}.pth")
    
    if not os.path.exists(model_path):
        print(f"Model {shadow_id} not found. Skipping.")
        return None

    model.load_state_dict(torch.load(model_path))
    model.eval()

    def get_confidences(loader, member_label):
        probabilities, labels, members = [], [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()  # Alle Klassenwahrscheinlichkeiten
                probabilities.append(probs)
                labels.append(targets.numpy())
                members.extend([member_label] * len(targets))
        
        return np.vstack(probabilities), np.hstack(labels), np.array(members)

    train_conf, train_labels, train_members = get_confidences(train_loader, member_label=1)
    test_conf, test_labels, test_members = get_confidences(test_loader, member_label=0)

    # Debugging: Überprüfung der Dimensionen
    print(f"Shadow Model {shadow_id} - Train conf shape: {train_conf.shape}, Test conf shape: {test_conf.shape}")

    # Konkateniere Trainings- und Testdaten
    confidences = np.concatenate((train_conf, test_conf), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    members = np.concatenate((train_members, test_members), axis=0)

    return confidences, labels, members

# Main function to process all shadow models and combine attack data
def main():
    all_probabilities, all_labels, all_members = [], [], []

    for shadow_id in range(20):  # 20 shadow models
        train_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/train/train.npz")
        test_path = os.path.join(SHADOW_DATA_DIR, f"shadow_model_{shadow_id}/test/test.npz")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Data for Shadow Model {shadow_id} not found. Skipping.")
            continue

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

    # Alle Wahrscheinlichkeiten in ein 2D-Array umwandeln
    all_probabilities = np.vstack(all_probabilities)  # Shape (N, 10)
    all_labels = np.hstack(all_labels)  # Shape (N,)
    all_members = np.hstack(all_members)  # Shape (N,)

    print(f"Final dataset contains {all_probabilities.shape[0]} samples.")
    print(f"First 5 probability vectors:\n{all_probabilities[:5]}")
    print(f"First 5 labels: {all_labels[:5]}")
    print(f"First 5 membership labels: {all_members[:5]}")

    # Split into train and test for the attacker model
    X_train, X_test, y_train, y_test = train_test_split(
        np.hstack((all_probabilities, all_labels.reshape(-1, 1))),  # Wahrscheinlichkeiten + Labels
        all_members,  # Ziel: Member/Non-Member
        test_size=0.3,
        random_state=42,
        stratify=all_members
    )

    print(f"Train Data Shape: {X_train.shape}, Test Data Shape: {X_test.shape}")

    # Save combined data
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
