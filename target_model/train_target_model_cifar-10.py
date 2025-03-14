# Train the Target Model on Cifar-10 Dataset

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

data_path = "~/amelie/data"


batch_size = 128
num_workers = 16
learning_rate = 0.001
num_epochs = 100
lr_decay = 1e-7  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#preprocessing =  Convert images to tensors and normalize pixel values to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load cifar-10 training and test data
train_dataset = datasets.CIFAR10(data_path, train=True, transform=transform, download=False)
test_dataset = datasets.CIFAR10(data_path, train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#simple model architecture (according to paper)
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # First conv layer (3 input channels -> 32 filters)
            nn.Tanh(),
            nn.MaxPool2d(2, 2), #downsampling
            nn.Conv2d(32, 64, kernel_size=3, padding=1), #second conv layer (32 filters -> 64 filters)
            nn.Tanh(),
            nn.MaxPool2d(2, 2) #downsampling
        )

        #fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),#flatten of feature maps into a single vector
            nn.Linear(64 * 8 * 8, 128), #first fully connected layer
            nn.Tanh(),
            nn.Linear(128, 10), #10 output classes for cifar-10
            nn.Softmax(dim=1) #logits -> probabilities
        )

    def forward(self, x):
        x = self.conv_layers(x) # Pass input through convolutional layers
        x = self.fc_layers(x) # Pass through fully connected layers
        return x
    

model = CIFAR10Model().to(device)

criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #adaptive learning rate optimizer

# Scheduler  -> When loss stagnates
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# safe training + validation accuracies and train_losses for plots
train_losses, test_accuracies, train_accuracies = [], [], []

#Training Loop
for epoch in range(num_epochs):
    model.train() #set model to training mode
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() #reset gradients
        outputs = model(inputs) #forward pass
        loss = criterion(outputs, labels)
        loss.backward() #backpropagation
        optimizer.step() #update weights

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1) #get predicted class labels
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item() #count correct predictions

    #compute avg training loss + accuracy    
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    test_accuracies.append(test_acc)

    # Update learning rate scheduler based on training loss
    scheduler.step(train_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

# Safe plots and target model
output_path = "/home/lab24inference/amelie/target_model"
os.makedirs(output_path, exist_ok=True)

plt.figure(figsize=(10, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.savefig(os.path.join(output_path, "training_metrics_cifar10.png"))
plt.show()

# Final model
torch.save(model.state_dict(), os.path.join(output_path, "cifar10_model.pth"))
print(f"Training done, model safed at {output_path}")
