import os
import numpy as np

# Verzeichnis mit den Shadow Model Daten
shadow_data_dir = "./shadow_data/"
num_shadow_models = 8  # Anzahl der Shadow Models

def load_data(path):
    """Hilfsfunktion, um Daten aus .npz-Dateien zu laden."""
    data = np.load(path)
    return data["images"], data["labels"]

# Lade alle Trainings- und Testdaten
train_data = []
test_data = []

for i in range(1, num_shadow_models + 1):
    train_path = os.path.join(shadow_data_dir, f"shadow_model_{i}/train/train_data.npz")
    test_path = os.path.join(shadow_data_dir, f"shadow_model_{i}/test/test_data.npz")
    
    images_train, labels_train = load_data(train_path)
    images_test, labels_test = load_data(test_path)
    
    train_data.append((images_train, labels_train))
    test_data.append((images_test, labels_test))

# Überprüfe auf Disjunktheit
def check_disjoint(data_list):
    """Prüft, ob die Daten disjunkt sind."""
    all_images = []
    all_labels = []
    for images, labels in data_list:
        all_images.append(images.reshape(len(images), -1))  # Flachstellen der Bilder
        all_labels.append(labels)

    # Kombiniere alle Daten
    all_images = np.vstack(all_images)
    all_labels = np.hstack(all_labels)
    
    # Erstelle einen Satz von Hashes für die Bilder
    image_hashes = set(map(hash, map(bytes, all_images)))
    if len(image_hashes) != len(all_images):
        print("Warnung: Überschneidungen in den Bildern gefunden!")
    else:
        print("Bilder sind disjunkt.")
    
    # Optional: Überprüfe die Labels, falls relevant
    if len(set(all_labels)) != len(all_labels):
        print("Warnung: Überschneidungen in den Labels gefunden!")
    else:
        print("Labels sind disjunkt.")

# Überprüfe die Trainings- und Testdaten
print("Überprüfung der Trainingsdaten:")
check_disjoint(train_data)

for i in range(8):
    for d in range (2):
        print(len(train_data[i][d]))
        print(len(test_data[i][d]))

print("\nÜberprüfung der Testdaten:")
check_disjoint(test_data)

 