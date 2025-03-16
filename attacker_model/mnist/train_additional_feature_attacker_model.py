import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import json

# Ensure directory for plots and model save path exist
MODEL_SAVE_DIR = "/home/lab24inference/amelie/attacker_model/mnist/models"
RESULTS_DIR = "/home/lab24inference/amelie/attacker_model/mnist/results"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)  # 

RESULTS_FILE = os.path.join(RESULTS_DIR, "additional_feature_attacker_model_results.json")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Load Data
DATA_FILE = "/home/lab24inference/amelie/shadow_models/mnist_models/attack_data/combined_attack_data.npz"
print(" Loading dataset...")
data = np.load(DATA_FILE)
X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.float32)
X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.float32)

# Compute additional features (Prediction Entropy and Gini Index)
def compute_extra_features(X):
    prediction_entropy = -np.sum(X[:, :10] * np.log(X[:, :10] + 1e-10), axis=1, keepdims=True)
    gini_index = 1 - np.sum(X[:, :10] ** 2, axis=1, keepdims=True)
    return np.hstack([prediction_entropy, gini_index])

extra_train_features = compute_extra_features(X_train)
extra_test_features = compute_extra_features(X_test)

X_train = np.hstack([X_train, extra_train_features])
X_test = np.hstack([X_test, extra_test_features])

print(f" Extended dataset shape: X_train {X_train.shape}, X_test {X_test.shape}")

# Define Dataset Class
class AttackerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)  # Send data to GPU if available
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)  # Send data to GPU if available
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define Attacker Model
class AttackerModel(nn.Module):
    def __init__(self, input_dim):
        super(AttackerModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.LeakyReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

# Training Function
def train_attacker_model(X_train, y_train, X_test, y_test):
    train_dataset = AttackerDataset(X_train, y_train)
    test_dataset = AttackerDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = AttackerModel(input_dim=X_train.shape[1]).to(device)  # Move model to GPU
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-5)
    
    # Tracking metrics
    train_losses, val_losses, accuracies, precisions, recalls, f1_scores, auc_scores = [], [], [], [], [], [], []

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Save training and validation metrics
        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                outputs = model(inputs).squeeze()
                test_preds.extend(outputs.cpu().numpy())  # Move back to CPU for evaluation
                test_targets.extend(labels.cpu().numpy())  # Move back to CPU for evaluation

        test_preds_binary = np.round(test_preds)
        accuracy = accuracy_score(test_targets, test_preds_binary)
        precision = precision_score(test_targets, test_preds_binary)
        recall = recall_score(test_targets, test_preds_binary)
        f1 = f1_score(test_targets, test_preds_binary)
        auc = roc_auc_score(test_targets, test_preds)

        # Append metrics to track
        train_losses.append(running_loss / len(train_loader))
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

        print(f"Epoch [{epoch+1}/20], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "additional_features_attacker_model.pth"))

    return {
        "train_losses": train_losses,
        "accuracies": accuracies,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "auc_scores": auc_scores
    }

## Final model training with features "Prediction Entropy + Gini Index + Class Label + Confidence Scores"
print(" Running final training...")

final_results = train_attacker_model(X_train[:, [11, 12, 0, 10]], y_train, X_test[:, [11, 12, 0, 10]], y_test)

# Save results
with open(RESULTS_FILE, "w") as json_file:
    json.dump(final_results, json_file, indent=4)


print(" Final model training completed and model saved.")
