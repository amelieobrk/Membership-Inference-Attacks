### Teile die synthetischen Daren, die ich mit meinen CGAN generiert habe in Trainings-und Testdaten für 
### 8 Shadow models auf


import os
import numpy as np
from sklearn.model_selection import train_test_split

# Pfade definieren
input_path = "./fake_batches"
output_path = "./shadow_data/"

# Zielanzahl der Shadow Models
num_shadow_models = 30

# Sicherstellen, dass der Ausgabepfad existiert
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Trainings- und Testdatenaufteilung
train_ratio = 0.5  # Verhältnis 50/50 damit das angriffsmodell anständig trainiert werden kann!

# Daten laden
data_files = [f for f in os.listdir(input_path) if f.endswith(".npz")]  # Lade alle Batches
all_images = []
all_labels = []

for file in data_files:
    data = np.load(os.path.join(input_path, file))
    all_images.append(data["images"])  # Annahme: 'images' Key existiert
    all_labels.append(data["labels"])  # Annahme: 'labels' Key existiert

# Alle Daten zusammenführen
all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Verifiziere, dass die Datenmenge durch die Anzahl der Shadow Models teilbar ist
assert len(all_images) % num_shadow_models == 0, "Die Datenmenge muss gleichmäßig auf die Shadow Models aufgeteilt werden können!"

# Daten in disjunkte Teile aufteilen
images_split = np.array_split(all_images, num_shadow_models) ## stelle sicher, dass alle Daten in disjunkte mengen aufgeteilt und in seperate orten gespeichert werden
labels_split = np.array_split(all_labels, num_shadow_models)

# Für jedes Shadow Model die Trainings- und Testdaten aufteilen
for i in range(num_shadow_models):
    # Split in Training und Test
    images_train, images_test, labels_train, labels_test = train_test_split(
        images_split[i], labels_split[i], test_size=(1 - train_ratio), random_state=i
    )

    # Shadow Model Ordner erstellen
    shadow_train_path = os.path.join(output_path, f"shadow_model_{i+1}/train")
    shadow_test_path = os.path.join(output_path, f"shadow_model_{i+1}/test")
    os.makedirs(shadow_train_path, exist_ok=True)
    os.makedirs(shadow_test_path, exist_ok=True)

    # Trainingsdaten speichern
    np.savez(os.path.join(shadow_train_path, f"train_data.npz"), images=images_train, labels=labels_train)

    # Testdaten speichern
    np.savez(os.path.join(shadow_test_path, f"test_data.npz"), images=images_test, labels=labels_test)

    print(f"Shadow Model {i+1}: Training und Testdaten gespeichert.")

print("Daten erfolgreich aufgeteilt!")
