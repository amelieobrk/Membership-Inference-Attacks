### split test und train ordner in 8 disjunkte datensätze

import os
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Pfade konfigurieren
BASE_DIR = os.path.expanduser("~/amelie/shadow_models_data/CINIC-10")
OUTPUT_DIR = os.path.expanduser("~/amelie/shadow_models_data/CINIC-10/shadow_data")
NUM_SHADOW_MODELS = 20

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Funktion zum Aufteilen der Daten

def split_data_per_class(input_dir, output_dir, num_splits):
    # Liste aller Klassen
    classes = os.listdir(input_dir)
    if not classes:
        raise ValueError("Keine Klassen im Verzeichnis gefunden!")
    
    for cls in classes:
        class_dir = os.path.join(input_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        # Alle Bilder in der Klasse auflisten
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.png')]
        
        # Daten gleichmäßig aufteilen
        splits = defaultdict(list)
        for i, img_path in enumerate(images):
            splits[i % num_splits].append(img_path)

        # Bilder in Shadow-Model-Ordner kopieren
        for i in range(num_splits):
            shadow_output_dir = os.path.join(output_dir, f"shadow_model_{i + 1}", cls)
            os.makedirs(shadow_output_dir, exist_ok=True)
            for img_path in splits[i]:
                shutil.copy(img_path, shadow_output_dir)

# Funktion zum Berechnen der Ordnergröße
def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# Funktion zum Zählen der Dateien in einem Ordner
def count_files_in_folder(folder):
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        total_files += len(filenames)
    return total_files

# Split für Training und Test
train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")

train_output_dir = os.path.join(OUTPUT_DIR, "train")
test_output_dir = os.path.join(OUTPUT_DIR, "test")

# Training und Testdaten splitten
split_data_per_class(train_dir, train_output_dir, NUM_SHADOW_MODELS)
split_data_per_class(test_dir, test_output_dir, NUM_SHADOW_MODELS)

# Ordnergrößen und Datei-Anzahlen ausgeben
def print_folder_info(base_output_dir, split_type):
    print(f"\n{split_type.capitalize()} Ordnerinformationen:")
    for i in range(NUM_SHADOW_MODELS):
        shadow_model_dir = os.path.join(base_output_dir, f"shadow_model_{i + 1}")
        size = get_folder_size(shadow_model_dir) / (1024 * 1024)  # Größe in MB
        file_count = count_files_in_folder(shadow_model_dir)  # Anzahl der Dateien
        print(f"Shadow Model {i + 1}: {size:.2f} MB, {file_count} Dateien")

print_folder_info(train_output_dir, "train")
print_folder_info(test_output_dir, "test")

print(f"Daten erfolgreich aufgeteilt in {NUM_SHADOW_MODELS} Shadow Models.")
