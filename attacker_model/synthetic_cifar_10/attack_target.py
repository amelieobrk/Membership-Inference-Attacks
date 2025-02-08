import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn as nn

# Pfad-Konfiguration
DATA_PATH = "~/amelie/data"
TARGET_MODEL_PATH = "/home/lab24inference/amelie/target_model/cifar10_model.pth"
ATTACKER_MODEL_PATH = "/home/lab24inference/amelie/attacker_model/synthetic_cifar_10/global_attacker_model.pth"
OUTPUT_PATH = "/home/lab24inference/amelie/attacker_model/synthetic_cifar_10/evaluation"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Hyperparameter
batch_size = 128
num_workers = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datenvorverarbeitung
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 Datensätze laden
train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, transform=transform, download=False)
test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Target Modell
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Modell initialisieren und laden
target_model = CIFAR10Model().to(device)
target_model.load_state_dict(torch.load(TARGET_MODEL_PATH))
target_model.eval()

# Wahrscheinlichkeitsausgaben und Labels sammeln
def get_outputs_and_labels(loader, model):
    probabilities = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(probabilities), np.hstack(labels)

print("Extrahiere Wahrscheinlichkeiten und Labels vom Target-Modell...")
train_probs, train_labels = get_outputs_and_labels(train_loader, target_model)
test_probs, test_labels = get_outputs_and_labels(test_loader, target_model)

print(f"Anzahl Member (Train): {len(train_probs)}")
print(f"Anzahl Non-Member (Test): {len(test_probs)}")

# Zusatzfeatures berechnen
def calculate_additional_features(probabilities):
    # Confidence Score: Höchste Wahrscheinlichkeit
    confidence_scores = np.max(probabilities, axis=1)

    # Prediction Entropy
    prediction_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

    # Prediction Confidence Gap: Unterschied zwischen den Top-2-Wahrscheinlichkeiten
    top_2_diff = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]

    # Standardabweichung der Wahrscheinlichkeiten
    std_dev = np.std(probabilities, axis=1)

    return confidence_scores, prediction_entropy, top_2_diff, std_dev

# Berechnung der zusätzlichen Features
confidence_scores, prediction_entropy, top_2_diffs, std_dev = calculate_additional_features(np.vstack((train_probs, test_probs)))

# Daten für das Attacker-Modell vorbereiten
train_membership = np.ones(len(train_probs))  # Trainingsdaten = Member
test_membership = np.zeros(len(test_probs))   # Testdaten = Non-Member

all_probs = np.vstack((train_probs, test_probs))
all_labels = np.hstack((train_labels, test_labels))
all_membership = np.hstack((train_membership, test_membership))

# Kombinierte Daten mit Zusatzfeatures
all_features = np.hstack((
    all_probs,
    confidence_scores.reshape(-1, 1),
    prediction_entropy.reshape(-1, 1),
    top_2_diffs.reshape(-1, 1),
    std_dev.reshape(-1, 1)
))

print("Shape of all_features:", all_features.shape)

# Attacker Modell
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

# Attacker Modell initialisieren und laden
attacker_model = AttackerModel(input_dim=all_features.shape[1]).to(device)
attacker_model.load_state_dict(torch.load(ATTACKER_MODEL_PATH))
attacker_model.eval()

# Evaluierung über alle Klassen
results = []
for class_id in range(10):
    print(f"Evaluierung für Klasse {class_id}...")

    class_indices = (all_labels == class_id)
    class_features = all_features[class_indices]
    class_membership = all_membership[class_indices]

    # Vorhersagen des Attacker-Modells
    def evaluate_attacker_model(features, membership, model):
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(features, dtype=torch.float32).to(device)
            membership_preds = model(inputs).squeeze().cpu().numpy()
            membership_preds_binary = (membership_preds >= 0.5).astype(int)

        precision = precision_score(membership, membership_preds_binary)
        recall = recall_score(membership, membership_preds_binary)
        f1 = f1_score(membership, membership_preds_binary)
        accuracy = accuracy_score(membership, membership_preds_binary)

        return precision, recall, f1, accuracy

    precision, recall, f1, accuracy = evaluate_attacker_model(class_features, class_membership, attacker_model)

    # Ergebnisse anzeigen und speichern
    print(f"Klasse {class_id}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    results.append((class_id, precision, recall, f1, accuracy))

# Ergebnisse speichern
results_path = os.path.join(OUTPUT_PATH, "attacker_model_evaluation_per_class.txt")
with open(results_path, "w") as f:
    for class_id, precision, recall, f1, accuracy in results:
        f.write(f"Klasse {class_id}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}\n")

print(f"Evaluation abgeschlossen. Ergebnisse unter: {results_path}")
