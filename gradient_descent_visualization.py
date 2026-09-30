import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter

print("="*60)
print("VISUALISATION DE LA DESCENTE DE GRADIENT")
print("Comparaison: Calcul Manuel vs PyTorch Automatique")
print("="*60)

# ============================================================
# Données d'entraînement simples
# ============================================================
# y = 3x + 2 + bruit
np.random.seed(42)
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
y = np.array([[5.0], [8.0], [11.0], [14.0], [17.0]], dtype=np.float32)

print("\nDonnées d'entraînement:")
for i in range(len(X)):
    print(f"  x={X[i][0]:.1f} -> y={y[i][0]:.1f}")

# ============================================================
# MÉTHODE 1: DESCENTE DE GRADIENT MANUELLE
# ============================================================
print("\n" + "="*60)
print("MÉTHODE 1: CALCUL MANUEL DES GRADIENTS")
print("="*60)

# Initialisation
w_manual = 0.0
b_manual = 0.0
learning_rate = 0.01
num_iterations = 100

# Historique pour visualisation
history_manual = {
    'w': [w_manual],
    'b': [b_manual],
    'loss': []
}

for iteration in range(num_iterations):
    # Forward pass: y_pred = w*x + b
    y_pred = w_manual * X + b_manual

    # Calcul de la loss (MSE)
    loss = np.mean((y_pred - y) ** 2)
    history_manual['loss'].append(loss)

    # Calcul MANUEL des gradients
    # dL/dw = (2/n) * sum((y_pred - y) * x)
    # dL/db = (2/n) * sum(y_pred - y)
    n = len(X)
    dL_dw = (2.0 / n) * np.sum((y_pred - y) * X)
    dL_db = (2.0 / n) * np.sum(y_pred - y)

    # Mise à jour des paramètres
    w_manual -= learning_rate * dL_dw
    b_manual -= learning_rate * dL_db

    history_manual['w'].append(w_manual)
    history_manual['b'].append(b_manual)

    if (iteration + 1) % 20 == 0:
        print(f"Iter {iteration+1}: Loss={loss:.4f}, w={w_manual:.4f}, b={b_manual:.4f}")

print(f"\nRésultat final (manuel):")
print(f"  w = {w_manual:.4f}")
print(f"  b = {b_manual:.4f}")

# ============================================================
# MÉTHODE 2: DESCENTE DE GRADIENT AVEC PYTORCH
# ============================================================
print("\n" + "="*60)
print("MÉTHODE 2: PYTORCH AUTOMATIQUE (loss.backward())")
print("="*60)

# Conversion en tensors PyTorch
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Initialisation (mêmes valeurs qu'en manuel)
w_torch = torch.tensor([[0.0]], requires_grad=True)
b_torch = torch.tensor([[0.0]], requires_grad=True)

# Historique
history_torch = {
    'w': [w_torch.item()],
    'b': [b_torch.item()],
    'loss': []
}

# TensorBoard
writer = SummaryWriter('runs/gradient_descent')

for iteration in range(num_iterations):
    # Forward pass
    y_pred = w_torch * X_tensor + b_torch

    # Calcul de la loss
    loss = torch.mean((y_pred - y_tensor) ** 2)
    history_torch['loss'].append(loss.item())

    # AUTOMATIQUE: Calcul des gradients avec backward()
    if w_torch.grad is not None:
        w_torch.grad.zero_()
    if b_torch.grad is not None:
        b_torch.grad.zero_()

    loss.backward()  # ← MAGIE DE PYTORCH!

    # Mise à jour manuelle (sans optimizer pour la comparaison)
    with torch.no_grad():
        w_torch -= learning_rate * w_torch.grad
        b_torch -= learning_rate * b_torch.grad

    history_torch['w'].append(w_torch.item())
    history_torch['b'].append(b_torch.item())

    # Log to TensorBoard
    writer.add_scalar('Loss/pytorch', loss.item(), iteration)
    writer.add_scalar('Parameters/w_pytorch', w_torch.item(), iteration)
    writer.add_scalar('Parameters/b_pytorch', b_torch.item(), iteration)

    if (iteration + 1) % 20 == 0:
        print(f"Iter {iteration+1}: Loss={loss.item():.4f}, w={w_torch.item():.4f}, b={b_torch.item():.4f}")

print(f"\nRésultat final (PyTorch):")
print(f"  w = {w_torch.item():.4f}")
print(f"  b = {b_torch.item():.4f}")

# ============================================================
# COMPARAISON
# ============================================================
print("\n" + "="*60)
print("COMPARAISON DES DEUX MÉTHODES")
print("="*60)
print(f"Différence w: {abs(w_manual - w_torch.item()):.6f}")
print(f"Différence b: {abs(b_manual - b_torch.item()):.6f}")
print("\n=> Les deux methodes donnent le meme resultat!")

# ============================================================
# VISUALISATIONS
# ============================================================
print("\n" + "="*60)
print("Création des visualisations...")
print("="*60)

# Figure 1: Évolution de la loss
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history_manual['loss'], 'b-', label='Manuel', linewidth=2)
ax1.plot(history_torch['loss'], 'r--', label='PyTorch', linewidth=2, alpha=0.7)
ax1.set_xlabel('Itération')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Évolution de la Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history_manual['loss'], 'b-', label='Manuel', linewidth=2)
ax2.plot(history_torch['loss'], 'r--', label='PyTorch', linewidth=2, alpha=0.7)
ax2.set_xlabel('Itération')
ax2.set_ylabel('Loss (MSE)')
ax2.set_title('Évolution de la Loss (échelle log)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_loss.png', dpi=150)
print("=> Sauvegarde: gradient_descent_loss.png")

# Figure 2: Évolution des paramètres
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history_manual['w'], 'b-', label='w (Manuel)', linewidth=2)
ax1.plot(history_torch['w'], 'r--', label='w (PyTorch)', linewidth=2, alpha=0.7)
ax1.axhline(y=3.0, color='g', linestyle=':', label='Valeur cible (3.0)')
ax1.set_xlabel('Itération')
ax1.set_ylabel('Valeur de w')
ax1.set_title('Convergence du Poids w')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history_manual['b'], 'b-', label='b (Manuel)', linewidth=2)
ax2.plot(history_torch['b'], 'r--', label='b (PyTorch)', linewidth=2, alpha=0.7)
ax2.axhline(y=2.0, color='g', linestyle=':', label='Valeur cible (2.0)')
ax2.set_xlabel('Itération')
ax2.set_ylabel('Valeur de b')
ax2.set_title('Convergence du Biais b')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_params.png', dpi=150)
print("=> Sauvegarde: gradient_descent_params.png")

# Figure 3: Surface de la loss et trajectoire de la descente
fig3 = plt.figure(figsize=(18, 6))

# Créer une grille pour la surface de loss
w_range = np.linspace(-1, 5, 100)
b_range = np.linspace(-2, 6, 100)
W_grid, B_grid = np.meshgrid(w_range, b_range)

# Calculer la loss pour chaque point de la grille
Loss_grid = np.zeros_like(W_grid)
for i in range(len(w_range)):
    for j in range(len(b_range)):
        y_pred_grid = W_grid[j, i] * X + B_grid[j, i]
        Loss_grid[j, i] = np.mean((y_pred_grid - y) ** 2)

# Vue 3D
ax1 = fig3.add_subplot(131, projection='3d')
surf = ax1.plot_surface(W_grid, B_grid, Loss_grid, cmap=cm.viridis, alpha=0.6)
# Utiliser les mêmes indices pour w, b et loss
ax1.plot(history_manual['w'][:-1], history_manual['b'][:-1], history_manual['loss'],
         'r-o', linewidth=2, markersize=3, label='Trajectoire')
ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('Loss')
ax1.set_title('Surface de Loss 3D')
ax1.view_init(elev=20, azim=45)
fig3.colorbar(surf, ax=ax1, shrink=0.5)

# Vue contour
ax2 = fig3.add_subplot(132)
contour = ax2.contour(W_grid, B_grid, Loss_grid, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(history_manual['w'], history_manual['b'], 'r-o', linewidth=2,
         markersize=4, label='Manuel')
ax2.plot(history_manual['w'][0], history_manual['b'][0], 'go',
         markersize=10, label='Départ')
ax2.plot(history_manual['w'][-1], history_manual['b'][-1], 'ro',
         markersize=10, label='Arrivée')
ax2.set_xlabel('w')
ax2.set_ylabel('b')
ax2.set_title('Trajectoire de la Descente de Gradient')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Vue contour avec heatmap
ax3 = fig3.add_subplot(133)
heatmap = ax3.contourf(W_grid, B_grid, Loss_grid, levels=50, cmap='viridis')
ax3.plot(history_manual['w'], history_manual['b'], 'r-', linewidth=2, alpha=0.8)
ax3.plot(history_manual['w'][::5], history_manual['b'][::5], 'ro',
         markersize=6, alpha=0.6)
ax3.plot(history_manual['w'][0], history_manual['b'][0], 'go',
         markersize=12, label='Départ', zorder=5)
ax3.plot(history_manual['w'][-1], history_manual['b'][-1], 'wo',
         markersize=12, label='Arrivée', zorder=5)
ax3.set_xlabel('w')
ax3.set_ylabel('b')
ax3.set_title('Heatmap de la Loss')
fig3.colorbar(heatmap, ax=ax3)
ax3.legend()

plt.tight_layout()
plt.savefig('gradient_descent_surface.png', dpi=150)
print("=> Sauvegarde: gradient_descent_surface.png")

# Figure 4: Comparaison des gradients
fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Recalculer les gradients manuels pour visualisation
gradients_w_manual = []
gradients_b_manual = []
w_temp = 0.0
b_temp = 0.0

for iteration in range(num_iterations):
    y_pred = w_temp * X + b_temp
    n = len(X)
    dL_dw = (2.0 / n) * np.sum((y_pred - y) * X)
    dL_db = (2.0 / n) * np.sum(y_pred - y)
    gradients_w_manual.append(dL_dw)
    gradients_b_manual.append(dL_db)
    w_temp -= learning_rate * dL_dw
    b_temp -= learning_rate * dL_db

# Gradients PyTorch
gradients_w_torch = []
gradients_b_torch = []
w_temp = torch.tensor([[0.0]], requires_grad=True)
b_temp = torch.tensor([[0.0]], requires_grad=True)

for iteration in range(num_iterations):
    y_pred = w_temp * X_tensor + b_temp
    loss = torch.mean((y_pred - y_tensor) ** 2)

    if w_temp.grad is not None:
        w_temp.grad.zero_()
        b_temp.grad.zero_()

    loss.backward()
    gradients_w_torch.append(w_temp.grad.item())
    gradients_b_torch.append(b_temp.grad.item())

    with torch.no_grad():
        w_temp -= learning_rate * w_temp.grad
        b_temp -= learning_rate * b_temp.grad

# Plot gradients
ax1.plot(gradients_w_manual, 'b-', label='Manuel', linewidth=2)
ax1.plot(gradients_w_torch, 'r--', label='PyTorch', linewidth=2, alpha=0.7)
ax1.set_xlabel('Itération')
ax1.set_ylabel('∂L/∂w')
ax1.set_title('Gradient par rapport à w')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(gradients_b_manual, 'b-', label='Manuel', linewidth=2)
ax2.plot(gradients_b_torch, 'r--', label='PyTorch', linewidth=2, alpha=0.7)
ax2.set_xlabel('Itération')
ax2.set_ylabel('∂L/∂b')
ax2.set_title('Gradient par rapport à b')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Magnitude des gradients
grad_magnitude_manual = [np.sqrt(gw**2 + gb**2) for gw, gb in zip(gradients_w_manual, gradients_b_manual)]
grad_magnitude_torch = [np.sqrt(gw**2 + gb**2) for gw, gb in zip(gradients_w_torch, gradients_b_torch)]

ax3.plot(grad_magnitude_manual, 'b-', label='Manuel', linewidth=2)
ax3.plot(grad_magnitude_torch, 'r--', label='PyTorch', linewidth=2, alpha=0.7)
ax3.set_xlabel('Itération')
ax3.set_ylabel('||∇L||')
ax3.set_title('Magnitude du Gradient (Norme)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Différence entre manuel et PyTorch
diff_w = [abs(m - t) for m, t in zip(gradients_w_manual, gradients_w_torch)]
diff_b = [abs(m - t) for m, t in zip(gradients_b_manual, gradients_b_torch)]

ax4.semilogy(diff_w, 'b-', label='Diff ∂L/∂w', linewidth=2)
ax4.semilogy(diff_b, 'r-', label='Diff ∂L/∂b', linewidth=2)
ax4.set_xlabel('Itération')
ax4.set_ylabel('Différence absolue (échelle log)')
ax4.set_title('Différence entre Manuel et PyTorch')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_gradients.png', dpi=150)
print("=> Sauvegarde: gradient_descent_gradients.png")

# Figure 5: Régression finale
fig5, ax = plt.subplots(1, 1, figsize=(10, 6))

x_plot = np.linspace(0, 6, 100)
y_manual_plot = w_manual * x_plot + b_manual
y_torch_plot = w_torch.item() * x_plot + b_torch.item()
y_true_plot = 3.0 * x_plot + 2.0

ax.scatter(X, y, color='black', s=100, label='Données', zorder=5)
ax.plot(x_plot, y_true_plot, 'g--', linewidth=2, label='Vraie fonction (y=3x+2)')
ax.plot(x_plot, y_manual_plot, 'b-', linewidth=2, label=f'Manuel (y={w_manual:.2f}x+{b_manual:.2f})')
ax.plot(x_plot, y_torch_plot, 'r--', linewidth=2, alpha=0.7, label=f'PyTorch (y={w_torch.item():.2f}x+{b_torch.item():.2f})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Régression Linéaire: Résultat Final')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_result.png', dpi=150)
print("=> Sauvegarde: gradient_descent_result.png")

# Add images to TensorBoard
writer.add_figure('Loss Evolution', fig1, 0)
writer.add_figure('Parameters Evolution', fig2, 0)
writer.add_figure('Loss Surface', fig3, 0)
writer.add_figure('Gradients Comparison', fig4, 0)
writer.add_figure('Final Result', fig5, 0)

writer.close()

print("\n" + "="*60)
print("VISUALISATIONS CRÉÉES:")
print("="*60)
print("  1. gradient_descent_loss.png - Évolution de la loss")
print("  2. gradient_descent_params.png - Convergence des paramètres")
print("  3. gradient_descent_surface.png - Surface de loss et trajectoire")
print("  4. gradient_descent_gradients.png - Comparaison des gradients")
print("  5. gradient_descent_result.png - Résultat final")
print("\nTensorBoard: runs/gradient_descent")
print("  Commande: tensorboard --logdir=runs")
print("  URL: http://localhost:6006")
print("="*60)

plt.show()
