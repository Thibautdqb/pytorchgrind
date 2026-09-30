import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# CNN Model for MNIST
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Helper function to show images
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Helper function to calculate accuracy
def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

print("="*60)
print("MNIST Image Classification with TensorBoard")
print("="*60)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# MNIST dataset
print("\nLoading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Initialize model
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard writer
writer = SummaryWriter('runs/mnist_classification')

# Get some random training images for visualization
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Create grid of images
img_grid = torchvision.utils.make_grid(images[:32])
writer.add_image('MNIST_images', img_grid)

# Add model graph to TensorBoard
writer.add_graph(model, images.to(device))

# Add embedding visualization
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# Select random images and their labels for embeddings
images_emb, labels_emb = select_n_random(train_dataset.data, train_dataset.targets, n=100)
images_emb = images_emb.unsqueeze(1).float() / 255.0  # Normalize to [0, 1]
images_emb = (images_emb - 0.5) / 0.5  # Normalize to [-1, 1]

# Get embeddings
class_labels = [str(i) for i in range(10)]
features = images_emb.view(100, -1)
writer.add_embedding(features,
                    metadata=labels_emb,
                    label_img=images_emb,
                    tag='mnist_embedding')

print("\n" + "="*60)
print("Training started...")
print("="*60)

# Training loop
total_step = len(train_loader)
global_step = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        # Log to TensorBoard every 100 steps
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = 100 * correct / total

            writer.add_scalar('Training/Loss', avg_loss, global_step)
            writer.add_scalar('Training/Accuracy', avg_acc, global_step)

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], '
                  f'Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%')

            running_loss = 0.0
            correct = 0
            total = 0

        global_step += 1

    # Evaluate on test set after each epoch
    test_acc = calculate_accuracy(model, test_loader, device)
    writer.add_scalar('Test/Accuracy', test_acc, epoch)

    # Log model parameters
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param, epoch)
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}] - Test Accuracy: {test_acc:.2f}%')
    print("-"*60)

# Final evaluation
print("\n" + "="*60)
print("Final Evaluation")
print("="*60)

final_train_acc = calculate_accuracy(model, train_loader, device)
final_test_acc = calculate_accuracy(model, test_loader, device)

print(f"Final Training Accuracy: {final_train_acc:.2f}%")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")

# Add PR curve for each class
class_probs = []
class_labels_list = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        class_probs.append(probs.cpu())
        class_labels_list.append(labels.cpu())

test_probs = torch.cat(class_probs)
test_labels = torch.cat(class_labels_list)

# Add PR curves for each class
for i in range(10):
    tensorboard_probs = test_probs[:, i]
    tensorboard_labels = (test_labels == i).int()
    writer.add_pr_curve(f'PR_Curve/class_{i}', tensorboard_labels, tensorboard_probs, global_step=0)

# Visualize some predictions
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Show 8 images with predictions
fig = plt.figure(figsize=(12, 4))
for idx in range(8):
    ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
    img = images[idx].cpu().squeeze()
    img = img / 2 + 0.5  # unnormalize
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Pred: {predicted[idx].item()}\nTrue: {labels[idx].item()}',
                color=("green" if predicted[idx]==labels[idx] else "red"))

writer.add_figure('Predictions', fig, global_step=0)

writer.close()

print("\n" + "="*60)
print("TensorBoard logs saved to 'runs/mnist_classification'")
print("TensorBoard is already running at http://localhost:6006")
print("Refresh the page to see the new MNIST classification results!")
print("\nIn TensorBoard, you can explore:")
print("  - SCALARS: Training/Test loss and accuracy curves")
print("  - IMAGES: Sample MNIST digits and predictions")
print("  - GRAPHS: CNN model architecture")
print("  - DISTRIBUTIONS/HISTOGRAMS: Weight and gradient evolution")
print("  - PROJECTOR: t-SNE/PCA embeddings of digits")
print("  - PR CURVES: Precision-Recall curves for each class")
print("="*60)
