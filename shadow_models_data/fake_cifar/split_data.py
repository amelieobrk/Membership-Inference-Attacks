### Split synthetic Cifar-10 data in test and train in a 50/50 Ratio for all shadow models


import os
import numpy as np
from sklearn.model_selection import train_test_split

# define paths
input_path = "./fake_batches"
output_path = "./shadow_data/"

num_shadow_models = 30

if not os.path.exists(output_path):
    os.makedirs(output_path)

#Train and Test Split
train_ratio = 0.5  # Ratio = 50/50 so that the attacker model gets an equal amount of members and non-members in its training and test data

# Load data
data_files = [f for f in os.listdir(input_path) if f.endswith(".npz")]  # Load all Batches
all_images = []
all_labels = []

for file in data_files:
    data = np.load(os.path.join(input_path, file))
    all_images.append(data["images"])  # 
    all_labels.append(data["labels"])  # 

# Combine all data
all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Verify that the amount of data is divisible by the number of shadow models
assert len(all_images) % num_shadow_models == 0, "It must be possible to distribute the data volume evenly across the shadow models!"

#  Divide data into disjoint parts
images_split = np.array_split(all_images, num_shadow_models) ## make sure that all data is split into disjoint sets and stored in separate locations
labels_split = np.array_split(all_labels, num_shadow_models)

# Split data in test and train for every shadow models
for i in range(num_shadow_models):

    images_train, images_test, labels_train, labels_test = train_test_split(
        images_split[i], labels_split[i], test_size=(1 - train_ratio), random_state=i
    )

    # Create Shadow model directories
    shadow_train_path = os.path.join(output_path, f"shadow_model_{i+1}/train")
    shadow_test_path = os.path.join(output_path, f"shadow_model_{i+1}/test")
    os.makedirs(shadow_train_path, exist_ok=True)
    os.makedirs(shadow_test_path, exist_ok=True)

    # Safe train data
    np.savez(os.path.join(shadow_train_path, f"train_data.npz"), images=images_train, labels=labels_train)

    # safe test data
    np.savez(os.path.join(shadow_test_path, f"test_data.npz"), images=images_test, labels=labels_test)

    print(f"Shadow model {i+1}: training and test data saved")

print("Data successfully split!")
