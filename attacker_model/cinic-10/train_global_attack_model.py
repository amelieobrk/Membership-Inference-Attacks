import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Konfiguration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/cinic_10_models/attack_data/combined_attack_data.npz"
MODEL_SAVE_DIR = "/home/lab24inference/amelie/attacker_model/cinic-10"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(MODEL_SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dataset-Klasse
class AttackerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Modell-Klasse
class AttackerModel(nn.Module):
    def __init__(self, input_dim):
        super(AttackerModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Focal Loss-Klasse
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Berechnung neuer kombinierter Features
def calculate_additional_features(probabilities):
    # Grundlegende Features
    confidence_scores = np.max(probabilities, axis=1)
    prediction_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    top_2_diff = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]

    # **Neue kombinierte Features**
    confidence_top2diff = confidence_scores * top_2_diff
    entropy_confidence = prediction_entropy * confidence_scores
    confidence_entropy_ratio = confidence_scores / (prediction_entropy + 1e-10)
    log_entropy = np.log(prediction_entropy + 1e-10)

    return confidence_top2diff, entropy_confidence, confidence_entropy_ratio, log_entropy

# Daten laden
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]

# Berechnung der neuen kombinierten Features
confidence_top2diff, entropy_confidence, confidence_entropy_ratio, log_entropy = calculate_additional_features(X_train[:, :-1])
X_train = np.hstack((confidence_top2diff.reshape(-1, 1), 
                     entropy_confidence.reshape(-1, 1), 
                     confidence_entropy_ratio.reshape(-1, 1), 
                     log_entropy.reshape(-1, 1)))

# Globales Modell trainieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split in Training und Testdaten
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# DataLoader erstellen
train_dataset = AttackerDataset(X_train_split, y_train_split)
val_dataset = AttackerDataset(X_val_split, y_val_split)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Modell initialisieren
model = AttackerModel(input_dim=X_train.shape[1]).to(device)
criterion = FocalLoss(alpha=2, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Training
epochs = 40
train_losses = []
val_losses = []
val_precisions = []
val_recalls = []
val_f1_scores = []
val_accuracies = []

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

    scheduler.step(train_loss)

    # Validierung
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_preds_binary = np.round(val_preds)
    precision = precision_score(val_targets, val_preds_binary)
    recall = recall_score(val_targets, val_preds_binary)
    f1 = f1_score(val_targets, val_preds_binary)
    accuracy = accuracy_score(val_targets, val_preds_binary)

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1_scores.append(f1)
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Plots erstellen
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
plt.title("Global Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "global_loss.png"))
plt.close()

plt.figure()
plt.plot(range(1, epochs + 1), val_precisions, label='Precision')
plt.plot(range(1, epochs + 1), val_recalls, label='Recall')
plt.plot(range(1, epochs + 1), val_f1_scores, label='F1 Score')
plt.plot(range(1, epochs + 1), val_accuracies, label='Accuracy')
plt.title("Global Model Metrics")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "global_metrics.png"))
plt.close()

print("Training completed.")
