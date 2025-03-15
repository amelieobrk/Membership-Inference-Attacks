import os
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import joblib

# Configuration
DATA_FILE = "/home/lab24inference/amelie/shadow_models/celebA_models/attack_data/combined_attack_data.npz"
MODEL_SAVE_DIR = "/home/lab24inference/amelie/attacker_model/celebA/models"
RESULTS_DIR = "/home/lab24inference/amelie/attacker_model/celebA/results"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
data = np.load(DATA_FILE)
X_train = data["X_train"][:, 0].reshape(-1, 1)  # Use only confidence scores
y_train = data["y_train"]
X_test = data["X_test"][:, 0].reshape(-1, 1)
y_test = data["y_test"]

# Choose the model: Random Forest (default) or Logistic Regression
use_random_forest = True  # Set to False to use Logistic Regression instead

if use_random_forest:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
else:
    model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_prob)

# Print results
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Save the results to a JSON file
results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "auc": auc
}

if use_random_forest:
    RESULTS_FILE = os.path.join(RESULTS_DIR, "random_forest.json")
else:
    RESULTS_FILE = os.path.join(RESULTS_DIR, "logistic_regression.json")
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {RESULTS_FILE}")

# Save the model
model_filename = "random_forest_model.pkl" if use_random_forest else "logistic_regression_model.pkl"
joblib.dump(model, os.path.join(MODEL_SAVE_DIR, model_filename))
print(f"Model saved as {model_filename}")
