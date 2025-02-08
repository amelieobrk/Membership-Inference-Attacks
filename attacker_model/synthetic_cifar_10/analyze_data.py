import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from collections import Counter

# Konfiguration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/synthetic_cifar_models/attack_data/combined_attack_data.npz"
ANALYSIS_DIR = "/home/lab24inference/amelie/attacker_model/synthetic_cifar_10/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Zusatzfeatures analysieren
def analyze_features(X, y):
    feature_names = ["Confidence Score", "Prediction Entropy", "Top-2 Diff", "Likelihood Ratio"]

    for i, feature_name in enumerate(feature_names):
        plt.figure()
        plt.hist(X[y == 1, i], bins=50, alpha=0.5, label="Members", color="blue", density=True)
        plt.hist(X[y == 0, i], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
        plt.title(f"Feature Distribution: {feature_name}")
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"feature_{feature_name.replace(' ', '_').lower()}_distribution.png"))
        plt.close()

# Klassenbalance überprüfen
def check_class_balance(y_train, y_val):
    train_counter = Counter(y_train)
    val_counter = Counter(y_val)

    print("Train Set Balance:", train_counter)
    print("Validation Set Balance:", val_counter)

    plt.figure()
    plt.bar(["Train Members", "Train Non-Members"], [train_counter[1], train_counter[0]], color=["blue", "red"])
    plt.bar(["Val Members", "Val Non-Members"], [val_counter[1], val_counter[0]], color=["blue", "red"], alpha=0.7)
    plt.title("Class Balance in Train and Validation Sets")
    plt.ylabel("Count")
    plt.savefig(os.path.join(ANALYSIS_DIR, "class_balance.png"))
    plt.close()

# Daten laden
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]

# Split in Training und Validierung
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# Zusatzfeatures extrahieren (Annahme: Zusatzfeatures sind die letzten Spalten)
additional_features = X_train[:, -4:]
analyze_features(additional_features, y_train)

# Klassenbalance überprüfen
check_class_balance(y_train_split, y_val_split)

print("Analysis completed. Results saved to:", ANALYSIS_DIR)
