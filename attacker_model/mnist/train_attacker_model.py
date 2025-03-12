import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/mnist_models/attack_data/combined_attack_data.npz"
MODEL_SAVE_DIR = "/home/lab24inference/amelie/attacker_model/mnist"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(MODEL_SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(MODEL_SAVE_DIR, "results.json")

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
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Load dataset
print(f"Loading dataset from: {DATA_FILE}")
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Ensure the input feature size matches
input_dim = X_train.shape[1]

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create DataLoader
train_dataset = AttackerDataset(X_train, y_train)
test_dataset = AttackerDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Initialize model, loss function, optimizer, and scheduler
model = AttackerModel(input_dim=input_dim).to(device)
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
epochs = 20
train_losses = []
test_losses = []
test_precisions = []
test_recalls = []
test_f1_scores = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    scheduler.step()

    # Evaluation
    model.eval()
    test_loss = 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_preds_binary = np.round(test_preds)
    precision = precision_score(test_targets, test_preds_binary, zero_division =0)
    recall = recall_score(test_targets, test_preds_binary, zero_division=0)
    f1 = f1_score(test_targets, test_preds_binary, zero_division=0)
    accuracy = accuracy_score(test_targets, test_preds_binary)

    train_losses.append(train_loss / len(train_loader))
    test_losses.append(test_loss / len(test_loader))
    test_precisions.append(precision)
    test_recalls.append(recall)
    test_f1_scores.append(f1)
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Save model
model_path = os.path.join(MODEL_SAVE_DIR, "global_attacker_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Global Attacker Model saved to {model_path}")

# Save results to JSON
results = {
    "train_losses": train_losses,
    "test_losses": test_losses,
    "test_precisions": test_precisions,
    "test_recalls": test_recalls,
    "test_f1_scores": test_f1_scores,
    "test_accuracies": test_accuracies
}
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {RESULTS_FILE}")

# Save plots
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.title("Global Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "global_loss.png"))
plt.close()

plt.figure()
plt.plot(range(1, epochs + 1), test_precisions, label='Precision')
plt.plot(range(1, epochs + 1), test_recalls, label='Recall')
plt.plot(range(1, epochs + 1), test_f1_scores, label='F1 Score')
plt.plot(range(1, epochs + 1), test_accuracies, label='Accuracy')
plt.title("Global Model Metrics")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "global_metrics.png"))
plt.close()

print("Training completed.")
