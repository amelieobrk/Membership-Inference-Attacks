import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, log_loss, roc_curve
from collections import Counter
from scipy.stats import entropy

# Configuration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/synthetic_cifar_models/attack_data/combined_attack_data.npz"
ANALYSIS_DIR = "/home/lab24inference/amelie/attacker_model/synthetic_cifar_10/attack_data_analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Define functions to compute additional metrices
def compute_top2_diff(probabilities):
    sorted_probs = np.sort(probabilities, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]

def compute_entropy(probabilities):
    return entropy(probabilities.T, base=2)


def compute_variance(probabilities):
    return np.var(probabilities, axis=1)

def compute_gini(probabilities):
    return 1 - np.sum(probabilities**2, axis=1)


# Load data from combined data file
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]
probability_matrix = X_train[:, :-1]


confidence_scores = np.max(probability_matrix, axis=1)
computed_top2_diff = compute_top2_diff(probability_matrix)
prediction_entropy = compute_entropy(probability_matrix)
variance_confidence = compute_variance(probability_matrix)
gini_index = compute_gini(probability_matrix)


# Plot results
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
plot_distribution(computed_top2_diff, y_train, "Top-2 Difference Distribution", "Top-2 Difference", "feature_top-2_diff_distribution.png")
plot_distribution(prediction_entropy, y_train, "Prediction Entropy Distribution", "Entropy", "feature_prediction_entropy_distribution.png")
plot_distribution(confidence_scores, y_train, "Feature Distribution: Confidence Score", "Confidence Score", "confidence_score_distribution.png")


# Check Class Balance
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
