import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, log_loss, roc_curve
from collections import Counter
from scipy.stats import entropy, ttest_ind

# Configuration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/mnist_models/attack_data/combined_attack_data.npz"
ANALYSIS_DIR = "/home/lab24inference/amelie/attacker_model/mnist/attack_data_analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Function to compute Top-2 Difference
def compute_top2_diff(probabilities):
    sorted_probs = np.sort(probabilities, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]

# Function to compute Prediction Entropy
def compute_entropy(probabilities):
    return entropy(probabilities.T, base=2)

# Function to compute Likelihood Ratio
def compute_likelihood_ratio(probabilities):
    sorted_probs = np.sort(probabilities, axis=1)
    return sorted_probs[:, -1] / (sorted_probs[:, -2] + 1e-10)  # Avoid division by zero

# Function to compute Variance of Confidence Scores
def compute_variance(probabilities):
    return np.var(probabilities, axis=1)

# Function to compute Gini Index
def compute_gini(probabilities):
    return 1 - np.sum(probabilities**2, axis=1)

# Function to compute Max-Min Confidence Spread
def compute_confidence_spread(probabilities):
    return np.max(probabilities, axis=1) - np.min(probabilities, axis=1)

# Load data
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]
probability_matrix = X_train[:, :-1]

# Compute additional metrics
computed_top2_diff = compute_top2_diff(probability_matrix)
prediction_entropy = compute_entropy(probability_matrix)
likelihood_ratio = compute_likelihood_ratio(probability_matrix)
variance_confidence = compute_variance(probability_matrix)
gini_index = compute_gini(probability_matrix)
confidence_spread = compute_confidence_spread(probability_matrix)

#Compute distribution of members and non members
plt.figure(figsize=(8, 6))
plt.bar(["Train Members", "Train Non-Members", "Test Members", "Test Non-Members"],
        [np.sum(y_train == 1), np.sum(y_train == 0), np.sum(data["y_test"] == 1), np.sum(data["y_test"] == 0)],
        color=["blue", "red", "blue", "red"], alpha=0.7)

plt.title("Distribution of Members and non members")
plt.ylabel("number")
plt.xlabel("class and dataset")

 #Plot
plt.savefig(os.path.join(ANALYSIS_DIR, "members_nonmembers_distribution.png"))
plt.show()


# Plot Variance of Confidence Scores Distribution
plt.figure(figsize=(8,6))
plt.hist(variance_confidence[y_train == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
plt.hist(variance_confidence[y_train == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
plt.title("Variance of Confidence Scores Distribution")
plt.xlabel("Variance")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, "variance_confidence_distribution.png"))
plt.show()

# Plot Gini Index Distribution
plt.figure(figsize=(8,6))
plt.hist(gini_index[y_train == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
plt.hist(gini_index[y_train == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
plt.title("Gini Index Distribution")
plt.xlabel("Gini Index")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, "gini_index_distribution.png"))
plt.show()

# Plot Max-Min Confidence Spread Distribution
plt.figure(figsize=(8,6))
plt.hist(confidence_spread[y_train == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
plt.hist(confidence_spread[y_train == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
plt.title("Max-Min Confidence Spread Distribution")
plt.xlabel("Confidence Spread")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, "confidence_spread_distribution.png"))
plt.show()

# Compute statistics
print("Statistics for Prediction Entropy:")
print("Mean (Members):", np.mean(prediction_entropy[y_train == 1]))
print("Mean (Non-Members):", np.mean(prediction_entropy[y_train == 0]))
print("Variance (Members):", np.var(prediction_entropy[y_train == 1]))
print("Variance (Non-Members):", np.var(prediction_entropy[y_train == 0]))

print("Statistics for Likelihood Ratio:")
print("Mean (Members):", np.mean(likelihood_ratio[y_train == 1]))
print("Mean (Non-Members):", np.mean(likelihood_ratio[y_train == 0]))
print("Variance (Members):", np.var(likelihood_ratio[y_train == 1]))
print("Variance (Non-Members):", np.var(likelihood_ratio[y_train == 0]))

print("Statistics for Variance of Confidence Scores:")
print("Mean (Members):", np.mean(variance_confidence[y_train == 1]))
print("Mean (Non-Members):", np.mean(variance_confidence[y_train == 0]))

print("Statistics for Gini Index:")
print("Mean (Members):", np.mean(gini_index[y_train == 1]))
print("Mean (Non-Members):", np.mean(gini_index[y_train == 0]))

print("Statistics for Confidence Spread:")
print("Mean (Members):", np.mean(confidence_spread[y_train == 1]))
print("Mean (Non-Members):", np.mean(confidence_spread[y_train == 0]))

print("Analysis completed. Results saved to:", ANALYSIS_DIR)