import os
import pandas as pd

#image directories
preprocessed_dir = "/home/lab24inference/amelie/data/preprocessed_celebA"
label_file = os.path.join(preprocessed_dir, "labels.csv")

# List for datanames and labels
data = []


for file_name in os.listdir(preprocessed_dir):
    if file_name.endswith(".jpg"):  # Nur Bilder berücksichtigen
        if file_name.startswith("smiling"):
            label = 1
        elif file_name.startswith("non_smiling"):
            label = 0
        else:
            continue  # Falls es eine unerwartete Datei gibt, überspringen

        data.append((file_name, label))

#Create Dataframe
df = pd.DataFrame(data, columns=["filename", "label"])

#Save CSV
df.to_csv(label_file, index=False)

print(f"Labels gespeichert in {label_file}")
