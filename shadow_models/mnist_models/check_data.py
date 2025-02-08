import numpy as np
import matplotlib.pyplot as plt
import os


###hier müssen noch die fertig trainierten models abgespeichert werden
def check_data(path, save_path):
    data = np.load(path)
    images = data['images']
    labels = data['labels']

    print("Pfad:", path)
    print("Bilder Dimensionen:", images.shape)
    print("Labels Dimensionen:", labels.shape)

    if images.ndim == 3:
        print("Bilder haben keine explizite Kanaldimension.")
    elif images.ndim == 4:
        print("Bilder haben eine explizite Kanaldimension.")
    else:
        print("Unerwartetes Format der Bilder.")

    plt.figure(figsize=(10, 2))
    for i in range(5):  # Nur die ersten 5 Bilder zeigen
        plt.subplot(1, 5, i + 1)
        if images.ndim == 4:
            plt.imshow(images[i].squeeze(), cmap='gray')
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    
    # Speichern des Plots statt Anzeigen
    plt.savefig(save_path)
    plt.close()
    print(f"Plot gespeichert unter: {save_path}")

def main():
    base_path = '/home/lab24inference/amelie/shadow_models_data/fake_mnist/'
    plot_base_path = '/home/lab24inference/amelie/shadow_models_data/fake_mnist/plots/'
    os.makedirs(plot_base_path, exist_ok=True)  # Stelle sicher, dass der Plot-Ordner existiert

    for i in range(20):  # Gehe von shadow_model_0 bis shadow_model_19
        train_path = os.path.join(base_path, f'shadow_model_{i}', 'train', 'train.npz')
        test_path = os.path.join(base_path, f'shadow_model_{i}', 'test', 'test.npz')
        train_plot_path = os.path.join(plot_base_path, f'shadow_model_{i}_train.png')
        test_plot_path = os.path.join(plot_base_path, f'shadow_model_{i}_test.png')
        
        print(f"Überprüfe Trainingsdaten für shadow_model_{i}")
        check_data(train_path, train_plot_path)
        print(f"Überprüfe Testdaten für shadow_model_{i}")
        check_data(test_path, test_plot_path)

if __name__ == '__main__':
    main()
