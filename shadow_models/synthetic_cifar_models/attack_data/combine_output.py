import os
import numpy as np
from sklearn.model_selection import train_test_split

# Verzeichnisse und Konfiguration
OUTPUT_DIR = "/home/lab24inference/amelie/shadow_models/synthetic_cifar_models/attack_data"
COMBINED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_attack_data.npz")

# find all shadow model files
shadow_files = [
    os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
    if f.startswith("shadow_model_") and f.endswith("_attack_data.npz")
]

# Initialize combined data
all_probabilities = []
all_labels = []
all_members = []

# load all data from all shadow models and combine them
for shadow_file in shadow_files:
    print(f"Load data from: {shadow_file}")
    data = np.load(shadow_file)
    all_probabilities.append(data["probabilities"])
    all_labels.append(data["labels"])
    all_members.append(data["members"])

# Kombinierte Daten erstellen
all_probabilities = np.vstack(all_probabilities)
all_labels = np.hstack(all_labels)
all_members = np.hstack(all_members)

print(f"Collected data points: {len(all_members)}")

# Split Train and test set for attacker model (30 / 70)
X_train, X_test, y_train, y_test = train_test_split(
    np.hstack((all_probabilities, all_labels.reshape(-1, 1))),  # Features= Probabilities + Labels
    all_members,  # goal: Member/Non-Member
    test_size=0.3,
    random_state=42,
    stratify=all_members  # Make sure member and non-member ratio stays the same
)

# Safe data
np.savez(
    COMBINED_OUTPUT_FILE,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print(f"Combined data path: {COMBINED_OUTPUT_FILE}")
print(f"Train Data: {X_train.shape}, Test Data: {X_test.shape}")
