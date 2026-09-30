import torch
from torchviz import make_dot

# Create simple training data
X_train_tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], requires_grad=False)
y_train_tensor = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]], requires_grad=False)

# Initialize parameters with gradient tracking
w = torch.tensor([[2.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

# Forward pass: simple linear regression y = b + w*X
y_pred = b + w * X_train_tensor

# Compute loss (MSE)
loss = ((y_pred - y_train_tensor) ** 2).mean()

# Create computational graph visualization
graph = make_dot(loss, params={'w': w, 'b': b})

print(f"Parameters:")
print(f"  w = {w.item()}")
print(f"  b = {b.item()}")
print(f"\nPredictions: {y_pred.squeeze().tolist()}")
print(f"Loss: {loss.item()}")

# Save DOT source
with open('linear_regression_graph.dot', 'w') as f:
    f.write(graph.source)
print(f"\nComputational graph saved to 'linear_regression_graph.dot'")

# Try to render PNG if Graphviz is available
try:
    graph.format = 'png'
    graph.render('linear_regression_graph', cleanup=True)
    print(f"PNG graph saved to 'linear_regression_graph.png'")
except Exception as e:
    print(f"\nNote: PNG rendering skipped (Graphviz not installed)")
    print("Install Graphviz from https://graphviz.org/download/ to generate PNG")

print("\n" + "="*60)
print("DOT Graph Source:")
print("="*60)
print(graph.source)
