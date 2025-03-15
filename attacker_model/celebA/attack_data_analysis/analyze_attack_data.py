import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Configuration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/celebA_models/attack_data/combined_attack_data.npz"
ANALYSIS_DIR = "/home/lab24inference/amelie/attacker_model/celebA/attack_data_analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Load data
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]

# Extract confidence scores (P(y=1))
confidence_scores = X_train[:, 0]  # Assuming first column contains P(y=1)

# Small epsilon for numerical stability
eps = 1e-10

# Compute additional metrics
variance_confidence = np.var(confidence_scores)
prediction_entropy = -confidence_scores * np.log2(confidence_scores + eps) - (1 - confidence_scores) * np.log2(1 - confidence_scores + eps)
gini_index = 2 * confidence_scores * (1 - confidence_scores)
log_odds = np.log((confidence_scores + eps) / (1 - confidence_scores + eps))

# Plot Confidence Score Distribution
plt.figure(figsize=(8,6))
plt.hist(confidence_scores[y_train == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
plt.hist(confidence_scores[y_train == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
plt.title("Confidence Score Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, "confidence_score_distribution.png"))
plt.show()

# Plot Variance of Confidence Scores
plt.figure(figsize=(8,6))
plt.hist(confidence_scores, bins=50, alpha=0.7, color="purple", density=True)
plt.title("Variance of Confidence Scores")
plt.xlabel("Variance")
plt.ylabel("Density")
plt.savefig(os.path.join(ANALYSIS_DIR, "variance_confidence_distribution.png"))
plt.show()

# Plot Prediction Entropy
plt.figure(figsize=(8,6))
plt.hist(prediction_entropy[y_train == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
plt.hist(prediction_entropy[y_train == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
plt.title("Prediction Entropy Distribution")
plt.xlabel("Entropy")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, "prediction_entropy_distribution.png"))
plt.show()

# Plot Gini Index
plt.figure(figsize=(8,6))
plt.hist(gini_index[y_train == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
plt.hist(gini_index[y_train == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
plt.title("Gini Index Distribution")
plt.xlabel("Gini Index")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, "gini_index_distribution.png"))
plt.show()

# Plot Log Odds
plt.figure(figsize=(8,6))
plt.hist(log_odds[y_train == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
plt.hist(log_odds[y_train == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
plt.title("Log Odds Distribution")
plt.xlabel("Log Odds")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, "log_odds_distribution.png"))
plt.show()

print("Analysis completed. Results saved to:", ANALYSIS_DIR)

import numpy as np

# Pfad zur .npz Datei
DATA_FILE = "/home/lab24inference/amelie/shadow_models/celebA_models/attack_data/combined_attack_data.npz"

# Laden der Daten
data = np.load(DATA_FILE)

# Alle gespeicherten Arrays in der Datei anzeigen
print("Keys in NPZ file:", data.files)

# Shape der gespeicherten Arrays ausgeben
for key in data.files:
    print(f"{key}: Shape = {data[key].shape}, Type = {data[key].dtype}")

# Beispielwerte ausgeben
print("\nBeispielwerte:")
for key in data.files:
    print(f"{key} (erste 5 Werte):\n", data[key][:5], "\n")
