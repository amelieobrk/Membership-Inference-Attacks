import os
import numpy as np
from sklearn.model_selection import train_test_split

# Verzeichnisse und Konfiguration
OUTPUT_DIR = "/home/lab24inference/amelie/shadow_models/synthetic_cifar_models/attack_data"
COMBINED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_attack_data.npz")

# Alle Shadow Model Dateien finden
shadow_files = [
    os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
    if f.startswith("shadow_model_") and f.endswith("_attack_data.npz")
]

# Initialisierung für die kombinierten Daten
all_probabilities = []
all_labels = []
all_members = []

# Daten aus allen Shadow Models laden und kombinieren
for shadow_file in shadow_files:
    print(f"Lade Daten aus: {shadow_file}")
    data = np.load(shadow_file)
    all_probabilities.append(data["probabilities"])
    all_labels.append(data["labels"])
    all_members.append(data["members"])

# Kombinierte Daten erstellen
all_probabilities = np.vstack(all_probabilities)
all_labels = np.hstack(all_labels)
all_members = np.hstack(all_members)

print(f"Gesamtanzahl Datenpunkte: {len(all_members)}")

# Trainings- und Testdaten aufteilen (70% Training, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(
    np.hstack((all_probabilities, all_labels.reshape(-1, 1))),  # Features: Wahrscheinlichkeiten + Labels
    all_members,  # Ziel: Member/Non-Member
    test_size=0.3,
    random_state=42,
    stratify=all_members  # Sicherstellen, dass Member/Non-Member-Verhältnis gleich bleibt
)

# Daten speichern
np.savez(
    COMBINED_OUTPUT_FILE,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print(f"Kombinierte Daten gespeichert unter: {COMBINED_OUTPUT_FILE}")
print(f"Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")
