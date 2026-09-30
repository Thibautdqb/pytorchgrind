import torch
import numpy as np

print("="*60)
print("PYTORCH SANS ML - EXEMPLES CONCRETS")
print("Manipulation et calcul de tenseurs")
print("="*60)

# ============================================================
# EXEMPLE 1: Calculs de physique - Trajectoire d'un projectile
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 1: TRAJECTOIRE D'UN PROJECTILE")
print("="*60)

# Paramètres
v0 = 20.0  # vitesse initiale (m/s)
angle = 45.0  # angle de tir (degrés)
g = 9.81  # gravité (m/s²)

# Temps de 0 à 4 secondes
t = torch.linspace(0, 4, 100)

# Conversion angle en radians
angle_rad = torch.tensor(angle * np.pi / 180)

# Calcul des composantes de vitesse
vx = v0 * torch.cos(angle_rad)
vy = v0 * torch.sin(angle_rad)

# Position x et y en fonction du temps
x = vx * t
y = vy * t - 0.5 * g * t**2

# Trouver la hauteur maximale et la portée
max_height = y.max()
max_height_time = t[y.argmax()]
range_x = x[y >= 0][-1]  # Dernière position où y >= 0

print(f"Vitesse initiale: {v0} m/s")
print(f"Angle: {angle}°")
print(f"Hauteur maximale: {max_height:.2f} m à t={max_height_time:.2f}s")
print(f"Portée: {range_x:.2f} m")

# ============================================================
# EXEMPLE 2: Traitement d'image - Convolution manuelle
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 2: CONVOLUTION D'IMAGE (filtre de détection de bords)")
print("="*60)

# Créer une "image" simple 5x5
image = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

# Filtre de Sobel pour détecter les bords verticaux
sobel_vertical = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=torch.float32)

print("Image originale:")
print(image)
print("\nFiltre Sobel (détection bords verticaux):")
print(sobel_vertical)

# Convolution manuelle (version simplifiée)
output_size = image.shape[0] - sobel_vertical.shape[0] + 1
result = torch.zeros(output_size, output_size)

for i in range(output_size):
    for j in range(output_size):
        patch = image[i:i+3, j:j+3]
        result[i, j] = (patch * sobel_vertical).sum()

print("\nRésultat de la convolution:")
print(result)

# ============================================================
# EXEMPLE 3: Statistiques - Calcul de corrélation
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 3: CORRELATION ENTRE DEUX VARIABLES")
print("="*60)

# Données: heures d'étude vs score à l'examen
heures_etude = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
scores = torch.tensor([50, 55, 60, 65, 70, 75, 78, 82, 85, 90], dtype=torch.float32)

# Calcul de la corrélation de Pearson
mean_x = heures_etude.mean()
mean_y = scores.mean()

# Écarts par rapport à la moyenne
x_centered = heures_etude - mean_x
y_centered = scores - mean_y

# Corrélation
numerator = (x_centered * y_centered).sum()
denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
correlation = numerator / denominator

print(f"Heures d'étude: {heures_etude.tolist()}")
print(f"Scores: {scores.tolist()}")
print(f"\nMoyenne heures: {mean_x:.2f}")
print(f"Moyenne scores: {mean_y:.2f}")
print(f"Corrélation de Pearson: {correlation:.4f}")
print(f"Interprétation: Corrélation {'forte' if correlation > 0.8 else 'modérée' if correlation > 0.5 else 'faible'} positive")

# ============================================================
# EXEMPLE 4: Algèbre linéaire - Résolution de système d'équations
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 4: SYSTEME D'EQUATIONS LINEAIRES")
print("="*60)

# Système:
# 2x + 3y = 13
# 4x - y = 5

A = torch.tensor([[2.0, 3.0],
                  [4.0, -1.0]])
b = torch.tensor([13.0, 5.0])

print("Système d'équations:")
print(f"  2x + 3y = 13")
print(f"  4x - y = 5")

# Résolution: x = A^(-1) * b
A_inv = torch.inverse(A)
solution = A_inv @ b

print(f"\nSolution:")
print(f"  x = {solution[0]:.2f}")
print(f"  y = {solution[1]:.2f}")

# Vérification
verification = A @ solution
print(f"\nVérification: A @ solution = {verification}")
print(f"Attendu: {b}")

# ============================================================
# EXEMPLE 5: Transformation géométrique - Rotation 2D
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 5: ROTATION DE POINTS 2D")
print("="*60)

# Points formant un carré
points = torch.tensor([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype=torch.float32)

# Angle de rotation (45 degrés)
angle = 45 * np.pi / 180

# Matrice de rotation
R = torch.tensor([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
], dtype=torch.float32)

# Appliquer la rotation
rotated_points = points @ R.T

print(f"Points originaux (carré):")
print(points)
print(f"\nAngle de rotation: 45°")
print(f"\nPoints après rotation:")
print(rotated_points)

# ============================================================
# EXEMPLE 6: Finance - Calcul de rendements et volatilité
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 6: ANALYSE DE SERIE TEMPORELLE FINANCIERE")
print("="*60)

# Prix d'une action sur 10 jours
prices = torch.tensor([100, 102, 101, 105, 103, 107, 106, 110, 108, 112], dtype=torch.float32)

# Calcul des rendements quotidiens (en %)
returns = (prices[1:] - prices[:-1]) / prices[:-1] * 100

# Statistiques
mean_return = returns.mean()
std_return = returns.std()
max_return = returns.max()
min_return = returns.min()

# Rendement cumulé
cumulative_return = (prices[-1] - prices[0]) / prices[0] * 100

print(f"Prix: {prices.tolist()}")
print(f"\nRendements quotidiens (%): {returns.tolist()}")
print(f"\nStatistiques:")
print(f"  Rendement moyen: {mean_return:.2f}%")
print(f"  Volatilité (écart-type): {std_return:.2f}%")
print(f"  Meilleur jour: +{max_return:.2f}%")
print(f"  Pire jour: {min_return:.2f}%")
print(f"  Rendement total: {cumulative_return:.2f}%")

# ============================================================
# EXEMPLE 7: Traitement de signal - Moyenne mobile
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 7: MOYENNE MOBILE (LISSAGE DE SIGNAL)")
print("="*60)

# Signal bruité
torch.manual_seed(42)
t = torch.linspace(0, 10, 100)
signal_clean = torch.sin(t)
noise = torch.randn(100) * 0.3
signal_noisy = signal_clean + noise

# Moyenne mobile sur fenêtre de 5
window_size = 5
signal_smoothed = torch.zeros(len(signal_noisy) - window_size + 1)

for i in range(len(signal_smoothed)):
    signal_smoothed[i] = signal_noisy[i:i+window_size].mean()

print(f"Signal original (10 premiers points): {signal_noisy[:10].tolist()}")
print(f"Signal lissé (10 premiers points): {signal_smoothed[:10].tolist()}")
print(f"\nRéduction du bruit:")
print(f"  Variance signal bruité: {signal_noisy.var():.4f}")
print(f"  Variance signal lissé: {signal_smoothed.var():.4f}")

# ============================================================
# EXEMPLE 8: Géométrie - Distance entre points
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 8: CALCUL DE DISTANCES (matrice de distances)")
print("="*60)

# 5 points en 2D
points = torch.tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, 0.5]
], dtype=torch.float32)

# Calcul de la matrice de distances (euclidienne)
n_points = points.shape[0]
distances = torch.zeros(n_points, n_points)

for i in range(n_points):
    for j in range(n_points):
        distances[i, j] = torch.sqrt(((points[i] - points[j])**2).sum())

print("Points:")
for i, p in enumerate(points):
    print(f"  P{i}: ({p[0]:.1f}, {p[1]:.1f})")

print("\nMatrice des distances:")
print(distances)

# Trouver les deux points les plus proches (hors diagonale)
mask = torch.eye(n_points, dtype=torch.bool)
distances_masked = distances.clone()
distances_masked[mask] = float('inf')
min_dist = distances_masked.min()
idx = (distances_masked == min_dist).nonzero()[0]

print(f"\nPaire la plus proche: P{idx[0].item()} et P{idx[1].item()}")
print(f"Distance: {min_dist:.4f}")

# ============================================================
# EXEMPLE 9: Batch processing - Normalisation de données
# ============================================================
print("\n" + "="*60)
print("EXEMPLE 9: NORMALISATION PAR BATCH")
print("="*60)

# Batch de données (4 samples, 3 features)
data = torch.tensor([
    [100, 20, 5],
    [150, 25, 8],
    [120, 22, 6],
    [180, 30, 10]
], dtype=torch.float32)

print("Données originales:")
print(data)

# Normalisation Min-Max (ramener dans [0, 1])
data_min = data.min(dim=0, keepdim=True).values
data_max = data.max(dim=0, keepdim=True).values
data_normalized_minmax = (data - data_min) / (data_max - data_min)

print("\nNormalisation Min-Max [0, 1]:")
print(data_normalized_minmax)

# Normalisation Z-score (moyenne=0, std=1)
data_mean = data.mean(dim=0, keepdim=True)
data_std = data.std(dim=0, keepdim=True)
data_normalized_zscore = (data - data_mean) / data_std

print("\nNormalisation Z-score (mean=0, std=1):")
print(data_normalized_zscore)
print(f"Moyenne après normalisation: {data_normalized_zscore.mean(dim=0)}")
print(f"Écart-type après normalisation: {data_normalized_zscore.std(dim=0)}")

print("\n" + "="*60)
print("FIN DES EXEMPLES")
print("="*60)
