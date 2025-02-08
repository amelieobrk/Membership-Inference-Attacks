### entferne alle cifar-10 daten und teile valid auf test und train auf


import os
import os
import shutil
import random
# Pfad zum CINIC-10-Datensatz
cinic10_path = "./"

def split_valid_into_train_and_test(valid_path, train_path, test_path):
    """Teilt die Dateien aus dem 'valid'-Ordner gleichmäßig auf 'train' und 'test' auf."""
    for class_folder in os.listdir(valid_path):
        class_path = os.path.join(valid_path, class_folder)
        train_class_path = os.path.join(train_path, class_folder)
        test_class_path = os.path.join(test_path, class_folder)

        # Sicherstellen, dass die Klassenordner existieren
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            random.shuffle(files)  # Zufällige Reihenfolge

            # Dateien gleichmäßig aufteilen
            split_point = len(files) // 2
            train_files = files[:split_point]
            test_files = files[split_point:]

            # Verschieben in train
            for file in train_files:
                shutil.move(os.path.join(class_path, file), os.path.join(train_class_path, file))

            # Verschieben in test
            for file in test_files:
                shutil.move(os.path.join(class_path, file), os.path.join(test_class_path, file))

            print(f"Klasse '{class_folder}': {len(train_files)} zu train, {len(test_files)} zu test verschoben.")


def count_files_in_folder(folder_path):
    """Zählt die Dateien in jedem Unterordner eines Ordners."""
    counts = {}
    for class_folder in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            counts[class_folder] = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    return counts

def remove_cifar10_files(folder_path):
    """Löscht alle Dateien, die mit 'cifar10-' beginnen, aus jedem Klassenordner."""
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.startswith("cifar10-"):
                    os.remove(os.path.join(class_path, file))

                    

# 1. Zählen der Dateien vor der Bereinigung
print("Anzahl der Dateien vor der Bereinigung:")

for folder in ["train", "valid", "test"]:
    folder_path = os.path.join(cinic10_path, folder)
    counts = count_files_in_folder(folder_path)
    print(f"{folder}: {counts}")

# 2. Bereinigung von CIFAR-10-Daten
print("\nBereinige Klassenordner von CIFAR-10-Daten...")
for folder in ["train", "valid", "test"]:
    folder_path = os.path.join(cinic10_path, folder)
    remove_cifar10_files(folder_path)

# 3. Zählen der Dateien nach der Bereinigung
print("\nAnzahl der Dateien nach der Bereinigung:")

for folder in ["train", "valid", "test"]:
    folder_path = os.path.join(cinic10_path, folder)
    counts = count_files_in_folder(folder_path)
    print(f"{folder}: {counts}")


# Pfade

valid_path = os.path.join(cinic10_path, "valid")
train_path = os.path.join(cinic10_path, "train")
test_path = os.path.join(cinic10_path, "test")

# Aufteilen
split_valid_into_train_and_test(valid_path, train_path, test_path)

# Entferne den leeren valid-Ordner
shutil.rmtree(valid_path)
print("\nValid-Ordner erfolgreich aufgeteilt und entfernt.")