### TODO: Feature probability dist (top 2) und unbedingt schauen ob alles richtig berechnet wurde


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, log_loss, roc_curve
from collections import Counter
from scipy.stats import entropy, ttest_ind
import seaborn as sns

# Configuration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/mnist_models/attack_data/combined_attack_data.npz"
ANALYSIS_DIR = "/home/lab24inference/amelie/attacker_model/mnist/attack_data_analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Function to analyze additional features
def analyze_features(X, y):
    available_features = X.shape[1]  # Get the actual number of features
    feature_names = ["Confidence Score", "Prediction Entropy", "Top-2 Diff", "Likelihood Ratio"][:available_features]
    
    print(f"Detected {available_features} additional features. Analyzing: {feature_names}")
    
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

# Function to check class balance and total counts
def check_class_balance(y_train, y_val):
    train_counter = Counter(y_train)
    val_counter = Counter(y_val)

    print("Train Set Balance:", train_counter)
    print("Validation Set Balance:", val_counter)
    print(f"Total Members: {train_counter[1] + val_counter[1]}")
    print(f"Total Non-Members: {train_counter[0] + val_counter[0]}")
    
    plt.figure()
    plt.bar(["Train Members", "Train Non-Members"], [train_counter[1], train_counter[0]], color=["blue", "red"])
    plt.bar(["Val Members", "Val Non-Members"], [val_counter[1], val_counter[0]], color=["blue", "red"], alpha=0.7)
    plt.title("Class Balance in Train and Validation Sets")
    plt.ylabel("Count")
    plt.savefig(os.path.join(ANALYSIS_DIR, "class_balance.png"))
    plt.close()

# Function to evaluate probability distributions across all classes
def evaluate_probability_distribution(X, y):
    prediction_entropy = np.apply_along_axis(entropy, 1, X)  # Compute entropy per sample
    mean_entropy_members = prediction_entropy[y == 1].mean()
    mean_entropy_nonmembers = prediction_entropy[y == 0].mean()
    print(f"Mean Prediction Entropy (Members): {mean_entropy_members:.4f}")
    print(f"Mean Prediction Entropy (Non-Members): {mean_entropy_nonmembers:.4f}")
    
    # Statistical significance test
    t_stat, p_value = ttest_ind(prediction_entropy[y == 1], prediction_entropy[y == 0], equal_var=False)
    print(f"T-Test: t={t_stat:.4f}, p={p_value:.4f}")
    
    plt.figure()
    plt.hist(prediction_entropy[y == 1], bins=50, alpha=0.5, label="Members", color="blue", density=True)
    plt.hist(prediction_entropy[y == 0], bins=50, alpha=0.5, label="Non-Members", color="red", density=True)
    plt.title("Prediction Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(ANALYSIS_DIR, "prediction_entropy_distribution.png"))
    plt.close()

# Load data
data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]

# Split into training and validation
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# Select 10 random indices for members and non-members
member_indices = np.random.choice(np.where(y_train == 1)[0], 10, replace=False)
nonmember_indices = np.random.choice(np.where(y_train == 0)[0], 10, replace=False)

# Print 10 member feature vectors
print("Sample Member Feature Vectors:")
print(X_train[member_indices])

# Print 10 non-member feature vectors
print("\nSample Non-Member Feature Vectors:")
print(X_train[nonmember_indices])

# Extract additional features dynamically
additional_features = X_train[:, -min(4, X_train.shape[1]):]  # Ensure we do not exceed the actual number of columns
analyze_features(additional_features, y_train)

# Check class balance
check_class_balance(y_train_split, y_val_split)

# Evaluate probability distributions
evaluate_probability_distribution(X_train, y_train)

# Compute additional MIA-related metrics
mean_confidence_member = X_train[y_train == 1, 0].mean()
mean_confidence_nonmember = X_train[y_train == 0, 0].mean()
roc_auc = roc_auc_score(y_train, X_train[:, 0])
logloss = log_loss(y_train, X_train[:, 0])

print(f"Mean Confidence Score (Members): {mean_confidence_member:.4f}")
print(f"Mean Confidence Score (Non-Members): {mean_confidence_nonmember:.4f}")
print(f"AUC-ROC for Confidence Score: {roc_auc:.4f}")
print(f"Log-Loss for Confidence Score: {logloss:.4f}")

print("Analysis completed. Results saved to:", ANALYSIS_DIR)
