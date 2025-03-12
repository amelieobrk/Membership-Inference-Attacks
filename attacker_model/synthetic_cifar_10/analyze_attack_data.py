import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, log_loss, roc_curve
from collections import Counter
from scipy.stats import entropy

# Konfiguration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/synthetic_cifar_models/attack_data/combined_attack_data.npz"
ANALYSIS_DIR = "/home/lab24inference/amelie/attacker_model/synthetic_cifar_10/attack_data_analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Funktionen zur Berechnung der Metriken
def compute_top2_diff(probabilities):
    sorted_probs = np.sort(probabilities, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]

def compute_entropy(probabilities):
    return entropy(probabilities.T, base=2)

def compute_likelihood_ratio(probabilities):
    sorted_probs = np.sort(probabilities, axis=1)
    return sorted_probs[:, -1] / (sorted_probs[:, -2] + 1e-10)

def compute_variance(probabilities):
    return np.var(probabilities, axis=1)

def compute_gini(probabilities):
    return 1 - np.sum(probabilities**2, axis=1)

def compute_confidence_spread(probabilities):
    return np.max(probabilities, axis=1) - np.min(probabilities, axis=1)

# Daten laden
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]
probability_matrix = X_train[:, :-1]

# Berechnung der Zusatzmetriken
computed_top2_diff = compute_top2_diff(probability_matrix)
prediction_entropy = compute_entropy(probability_matrix)
likelihood_ratio = compute_likelihood_ratio(probability_matrix)
variance_confidence = compute_variance(probability_matrix)
gini_index = compute_gini(probability_matrix)
confidence_spread = compute_confidence_spread(probability_matrix)

# Plots für jede Metrik
def plot_distribution(values, y_labels, title, xlabel, filename):
    plt.figure(figsize=(8,6))
    plt.hist(values[y_labels == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
    plt.hist(values[y_labels == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(ANALYSIS_DIR, filename))
    plt.close()

plot_distribution(variance_confidence, y_train, "Variance of Confidence Scores Distribution", "Variance", "variance_confidence_distribution.png")
plot_distribution(gini_index, y_train, "Gini Index Distribution", "Gini Index", "gini_index_distribution.png")
plot_distribution(confidence_spread, y_train, "Max-Min Confidence Spread Distribution", "Confidence Spread", "confidence_spread_distribution.png")
plot_distribution(computed_top2_diff, y_train, "Top-2 Difference Distribution", "Top-2 Difference", "feature_top-2_diff_distribution.png")
plot_distribution(prediction_entropy, y_train, "Prediction Entropy Distribution", "Entropy", "feature_prediction_entropy_distribution.png")
plot_distribution(likelihood_ratio, y_train, "Likelihood Ratio Distribution", "Likelihood Ratio", "feature_likelihood_ratio_distribution.png")

# Klassenbalance prüfen
def check_class_balance(y):
    counter = Counter(y)
    plt.figure()
    plt.bar(["Members", "Non-Members"], [counter[1], counter[0]], color=["blue", "red"])
    plt.title("Class Balance")
    plt.ylabel("Count")
    plt.savefig(os.path.join(ANALYSIS_DIR, "class_balance.png"))
    plt.close()
    print("Class Balance:", counter)

check_class_balance(y_train)

print("Analysis completed. Results saved to:", ANALYSIS_DIR)
