import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Create a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Create training data: y = 2x + 1
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

# Initialize model
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create TensorBoard writer
writer = SummaryWriter('runs/linear_regression')

# Add the model graph to TensorBoard
writer.add_graph(model, X_train)

print("Training Linear Regression Model...")
print("="*60)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

    # Log weights and biases
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        writer.add_scalar(f'Parameters/{name}', param.mean(), epoch)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Final predictions
with torch.no_grad():
    y_pred_final = model(X_train)
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    print(f"Weight: {model.linear.weight.item():.4f}")
    print(f"Bias: {model.linear.bias.item():.4f}")
    print(f"\nPredictions vs Actual:")
    for i in range(len(X_train)):
        print(f"  x={X_train[i].item():.1f} -> pred={y_pred_final[i].item():.4f}, actual={y_train[i].item():.1f}")

# Add final predictions to TensorBoard
writer.add_text('Model Parameters',
                f'Weight: {model.linear.weight.item():.4f}, Bias: {model.linear.bias.item():.4f}')

writer.close()

print("\n" + "="*60)
print("TensorBoard logs saved to 'runs/linear_regression'")
print("To view in TensorBoard, run:")
print("  tensorboard --logdir=runs")
print("Then open http://localhost:6006 in your browser")
print("="*60)
