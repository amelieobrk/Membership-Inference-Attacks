#train a cnn on the celebA Data using prediction entropy and log odds as additional features

import os
import numpy as np
import torch
import json
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

# Configuration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/celebA_models/attack_data/combined_attack_data.npz"
MODEL_SAVE_DIR = "/home/lab24inference/amelie/attacker_model/celebA/models"
RESULTS_DIR = "/home/lab24inference/amelie/attacker_model/celebA/results"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "attacker_model_results.json")

# Lade Daten
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Extract Confidence Score (P(y=1))
confidence_scores_train = X_train[:, 0]
confidence_scores_test = X_test[:, 0]

# Epsilon for numeric stability
eps = 1e-10

# compute additional features
prediction_entropy_train = -confidence_scores_train * np.log2(confidence_scores_train + eps) - (1 - confidence_scores_train) * np.log2(1 - confidence_scores_train + eps)
log_odds_train = np.log((confidence_scores_train + eps) / (1 - confidence_scores_train + eps))

prediction_entropy_test = -confidence_scores_test * np.log2(confidence_scores_test + eps) - (1 - confidence_scores_test) * np.log2(1 - confidence_scores_test + eps)
log_odds_test = np.log((confidence_scores_test + eps) / (1 - confidence_scores_test + eps))

# new feature set for attacker model
X_train_attacker = np.column_stack((confidence_scores_train, prediction_entropy_train, log_odds_train))
X_test_attacker = np.column_stack((confidence_scores_test, prediction_entropy_test, log_odds_test))

# Membership-Labels stay the same!
y_train_attacker = y_train
y_test_attacker = y_test

# Dataset class
class AttackerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Attacker Model
class AttackerModel(nn.Module):
    def __init__(self, input_dim):
        super(AttackerModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

input_dim = X_train_attacker.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = AttackerDataset(X_train_attacker, y_train_attacker)
test_dataset = AttackerDataset(X_test_attacker, y_test_attacker)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Initialize model, loss function, optimizer
model = AttackerModel(input_dim=input_dim).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
epochs = 1
test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_preds_binary = np.round(test_preds)
    accuracy = accuracy_score(test_targets, test_preds_binary)
    precision = precision_score(test_targets, test_preds_binary, zero_division=0)
    recall = recall_score(test_targets, test_preds_binary, zero_division=0)
    f1 = f1_score(test_targets, test_preds_binary, zero_division=0)
    auc = roc_auc_score(test_targets, test_preds)
    
    test_metrics["accuracy"].append(accuracy)
    test_metrics["precision"].append(precision)
    test_metrics["recall"].append(recall)
    test_metrics["f1"].append(f1)
    test_metrics["auc"].append(auc)
    
    print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Save the results to a JSON file
results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "auc": auc
}

RESULTS_FILE = os.path.join(RESULTS_DIR, "cnn.json")
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)

# Save model
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "cnn.pth"))
print("Attacker Model saved!")
