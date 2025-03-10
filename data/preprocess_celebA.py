import pandas as pd
from PIL import Image
import os

src_dir = '/home/lab24inference/amelie/data/celebA'  
attr_file = os.path.join(src_dir, 'list_attr_celeba.txt')
dest_dir = '/home/lab24inference/amelie/data/preprocessed_celebA'
sample_size = 25000  


if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

#load attributes 
attrs = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)
attrs.replace(-1, 0, inplace=True)  

#filter smiling and non-smiling 25.000 images each
smiling = attrs[attrs['Smiling'] == 1].sample(sample_size, random_state=42)
non_smiling = attrs[attrs['Smiling'] == 0].sample(sample_size, random_state=42)

#process images from txt file and resize pcitures to 64x64 pixels 
def process_and_save_images(df, category):
    count = 0
    for index, row in df.iterrows():
        file_name = index 
        file_path = os.path.join(src_dir, file_name)

        if not os.path.exists(file_path):  
            print(f"Warnung: Datei {file_path} existiert nicht, überspringe...")
            continue  

        image = Image.open(file_path)
        image = image.resize((64, 64), Image.ANTIALIAS)
        save_path = os.path.join(dest_dir, f"{category}_{count}.jpg")
        image.save(save_path)
        count += 1

        if count % 1000 == 0:
            print(f"Processed {count} images for {category}")

print("Image processing complete.")



# Bilder verarbeiten und speichern
process_and_save_images(smiling, 'smiling')
process_and_save_images(non_smiling, 'non_smiling')

print("Image processing complete.")
