#Perform ablation study using different feature combinations

import os
import itertools
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import json

# Check CUDA availability

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

BASE_DIR = "/home/lab24inference/amelie/attacker_model/cifar_10/ablation_study"
RESULTS_FILE = os.path.join(BASE_DIR, "ablation_results.json")

# Load Data
DATA_FILE = "/home/lab24inference/amelie/shadow_models/cifar_models/attack_data/combined_attack_data.npz"
print(" Loading dataset...")
data = np.load(DATA_FILE)
X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.float32)
X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.float32)

print(f"[ Dataset loaded successfully: X_train {X_train.shape}, X_test {X_test.shape}")

# Compute additional features
def compute_extra_features(X):
    prediction_entropy = -np.sum(X[:, :10] * np.log(X[:, :10] + 1e-10), axis=1, keepdims=True)
    top_2_diff = np.sort(X[:, :10], axis=1)[:, -1:] - np.sort(X[:, :10], axis=1)[:, -2:-1]
    gini_index = 1 - np.sum(X[:, :10] ** 2, axis=1, keepdims=True)
    variance = np.var(X[:, :10], axis=1, keepdims=True)
    
    return np.hstack([prediction_entropy, top_2_diff, gini_index, variance])

extra_train_features = compute_extra_features(X_train)
extra_test_features = compute_extra_features(X_test)

X_train = np.hstack([X_train, extra_train_features])
X_test = np.hstack([X_test, extra_test_features])

print(f"Extended dataset shape: X_train {X_train.shape}, X_test {X_test.shape}")

# Define Dataset Class
class AttackerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
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
            nn.Linear(16, 1), nn.Sigmoid()  # Binary classification output (0 or 1)
        )
    def forward(self, x):
        return self.fc(x)

# Training Function
def train_attacker_model(X_train, y_train, X_test, y_test):
    train_dataset = AttackerDataset(X_train, y_train)
    test_dataset = AttackerDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = AttackerModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-5)
    
    for epoch in range(20):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    test_preds_binary = np.round(test_preds)
    return {
        "Accuracy": accuracy_score(test_targets, test_preds_binary),
        "Precision": precision_score(test_targets, test_preds_binary),
        "Recall": recall_score(test_targets, test_preds_binary),
        "F1": f1_score(test_targets, test_preds_binary),
        "ROC AUC": roc_auc_score(test_targets, test_preds)
    }

# Load existing results
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as json_file:
        ablation_results = json.load(json_file)
else:
    ablation_results = {}

# Baseline Model Training
print(" Running baseline training (only with confidence scores and class labels)...")
baseline_results = train_attacker_model(X_train[:, :11], y_train, X_test[:, :11], y_test)  # only Confidence Scores and Class Labels
ablation_results["Baseline"] = {
    "Used Features": ["10 Confidence Scores", "Class Label"],
    "Results": baseline_results
}
with open(RESULTS_FILE, "w") as json_file:
    json.dump(ablation_results, json_file, indent=4)
print(f"[INFO] Baseline Results: {baseline_results}")

# Ablation Study without Confidence Scores (testing additional features alone)
EXTRA_FEATURES = ["Prediction Entropy", "Top-2 Difference", "Gini Index", "Variance"]
BASELINE_FEATURES = list(range(11))

print(" Running ablation study (training with additional features without Confidence Scores)...")
for num_add in range(1, len(EXTRA_FEATURES) + 1):
    for added_features in itertools.combinations(EXTRA_FEATURES, num_add):
        selected_feature_indices = [11 + EXTRA_FEATURES.index(f) for f in added_features]  # only select additional features
        X_train_reduced = X_train[:, selected_feature_indices]
        X_test_reduced = X_test[:, selected_feature_indices]
        
        print(f"[INFO] Training with features: { list(added_features)} (without Confidence Scores)")
        results = train_attacker_model(X_train_reduced, y_train, X_test_reduced, y_test)
        ablation_results[str(added_features)] = {
            "Used Features": list(added_features),
            "Results": results
        }
        with open(RESULTS_FILE, "w") as json_file:
            json.dump(ablation_results, json_file, indent=4)
        print(f"Results after adding {added_features}: {results}")

print(" Ablation study completed.")
