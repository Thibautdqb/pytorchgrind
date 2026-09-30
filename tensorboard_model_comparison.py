import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time

# ============================================================
# MODÈLE 1: Ultra Simple - Juste une couche linéaire
# ============================================================
class Model1_Simple(nn.Module):
    def __init__(self):
        super(Model1_Simple, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # Direct: pixels -> classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc(x)
        return x

# ============================================================
# MODÈLE 2: Une couche cachée
# ============================================================
class Model2_OneHidden(nn.Module):
    def __init__(self):
        super(Model2_OneHidden, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================
# MODÈLE 3: Deux couches cachées
# ============================================================
class Model3_TwoHidden(nn.Module):
    def __init__(self):
        super(Model3_TwoHidden, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============================================================
# MODÈLE 4: Avec Dropout (pour éviter l'overfitting)
# ============================================================
class Model4_WithDropout(nn.Module):
    def __init__(self):
        super(Model4_WithDropout, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ============================================================
# MODÈLE 5: CNN Simple (utilise la structure spatiale)
# ============================================================
class Model5_CNN(nn.Module):
    def __init__(self):
        super(Model5_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# Fonction d'entraînement
# ============================================================
def train_model(model, model_name, train_loader, test_loader, device, num_epochs=5):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(f'runs/comparison/{model_name}')

    # Add model graph
    images, _ = next(iter(train_loader))
    writer.add_graph(model, images.to(device))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total

        # Log to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        # Log weight distributions
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%')

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f}s")

    # Final test accuracy
    final_test_acc = test_acc
    writer.add_text('Summary',
                   f'Parameters: {total_params:,}\n'
                   f'Final Test Accuracy: {final_test_acc:.2f}%\n'
                   f'Training Time: {training_time:.2f}s')

    writer.close()
    return final_test_acc

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("COMPARAISON DE MODÈLES AVEC TENSORBOARD")
    print("Du plus simple au plus sophistiqué")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load MNIST
    print("\nChargement de MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Données d'entraînement: {len(train_dataset)}")
    print(f"Données de test: {len(test_dataset)}")

    # Define models to compare
    models = [
        (Model1_Simple().to(device), "1_Simple_Linear"),
        (Model2_OneHidden().to(device), "2_One_Hidden_Layer"),
        (Model3_TwoHidden().to(device), "3_Two_Hidden_Layers"),
        (Model4_WithDropout().to(device), "4_With_Dropout"),
        (Model5_CNN().to(device), "5_CNN"),
    ]

    results = []

    # Train each model
    for model, name in models:
        acc = train_model(model, name, train_loader, test_loader, device, num_epochs=5)
        results.append((name, acc))

    # Print summary
    print("\n" + "="*60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*60)
    for name, acc in results:
        print(f"{name:30s} -> Test Accuracy: {acc:.2f}%")

    print("\n" + "="*60)
    print("Visualisez les résultats dans TensorBoard:")
    print("  http://localhost:6006")
    print("\nDans TensorBoard, vous pouvez:")
    print("  - SCALARS: Comparer les courbes de loss et accuracy")
    print("  - GRAPHS: Voir l'architecture de chaque modèle")
    print("  - HISTOGRAMS: Observer l'évolution des poids")
    print("\nUtilisez le sélecteur de runs en bas à gauche pour")
    print("comparer les différents modèles côte à côte!")
    print("="*60)

if __name__ == "__main__":
    main()
