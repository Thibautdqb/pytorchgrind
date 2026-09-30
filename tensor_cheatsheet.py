import torch
import numpy as np

print("="*70)
print(" " * 20 + "PYTORCH TENSOR CHEATSHEET")
print("="*70)

# ============================================================
# 1. CREATION DE TENSEURS
# ============================================================
print("\n" + "="*70)
print("1. CREATION DE TENSEURS")
print("="*70)

# Depuis une liste Python
t1 = torch.tensor([1, 2, 3])
print(f"Depuis liste: {t1}")

# Depuis numpy
arr = np.array([1, 2, 3])
t2 = torch.from_numpy(arr)
print(f"Depuis numpy: {t2}")

# Tenseurs de zéros et de uns
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
print(f"\nZeros (2x3):\n{zeros}")
print(f"Ones (2x3):\n{ones}")

# Tenseur rempli d'une valeur
filled = torch.full((2, 3), 7.5)
print(f"Rempli de 7.5:\n{filled}")

# Tenseur aléatoire
random_uniform = torch.rand(2, 3)  # Uniforme [0, 1]
random_normal = torch.randn(2, 3)  # Normal(0, 1)
print(f"\nAléatoire uniforme:\n{random_uniform}")
print(f"Aléatoire normal:\n{random_normal}")

# Tenseur d'entiers aléatoires
random_int = torch.randint(0, 10, (2, 3))  # Entre 0 et 10
print(f"Entiers aléatoires [0, 10):\n{random_int}")

# Séquences
arange = torch.arange(0, 10, 2)  # De 0 à 10, pas de 2
linspace = torch.linspace(0, 1, 5)  # 5 valeurs entre 0 et 1
print(f"\narange(0, 10, 2): {arange}")
print(f"linspace(0, 1, 5): {linspace}")

# Matrice identité
identity = torch.eye(3)
print(f"\nMatrice identité:\n{identity}")

# Tenseur comme un autre (même forme)
x = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x, dtype=torch.float32)
print(f"\nOriginal:\n{x}")
print(f"zeros_like:\n{zeros_like}")
print(f"rand_like:\n{rand_like}")

# ============================================================
# 2. PROPRIETES DES TENSEURS
# ============================================================
print("\n" + "="*70)
print("2. PROPRIETES DES TENSEURS")
print("="*70)

t = torch.randn(2, 3, 4)
print(f"Tenseur de forme {t.shape}")
print(f"  .shape ou .size(): {t.shape}")
print(f"  .ndim (dimensions): {t.ndim}")
print(f"  .numel() (nb d'elements): {t.numel()}")
print(f"  .dtype (type): {t.dtype}")
print(f"  .device (cpu/gpu): {t.device}")
print(f"  .requires_grad: {t.requires_grad}")

# ============================================================
# 3. CHANGEMENT DE FORME (RESHAPE)
# ============================================================
print("\n" + "="*70)
print("3. CHANGEMENT DE FORME (RESHAPE)")
print("="*70)

x = torch.arange(12)
print(f"Original: {x}")

# Reshape
reshaped = x.reshape(3, 4)
print(f"\nreshape(3, 4):\n{reshaped}")

# View (comme reshape mais partage la mémoire)
viewed = x.view(2, 6)
print(f"view(2, 6):\n{viewed}")

# Utiliser -1 pour inférer automatiquement
auto = x.reshape(2, -1)  # Infère 6
print(f"reshape(2, -1):\n{auto}")

# Aplatir (flatten)
flattened = reshaped.flatten()
print(f"\nflatten(): {flattened}")

# Squeeze: supprimer dimensions de taille 1
squeezable = torch.zeros(2, 1, 3, 1)
squeezed = squeezable.squeeze()
print(f"\nsqueeze {squeezable.shape} -> {squeezed.shape}")

# Unsqueeze: ajouter une dimension
x = torch.tensor([1, 2, 3])
unsqueezed = x.unsqueeze(0)  # Ajoute dim au début
print(f"unsqueeze(0): {x.shape} -> {unsqueezed.shape}")
unsqueezed = x.unsqueeze(1)  # Ajoute dim à la fin
print(f"unsqueeze(1): {x.shape} -> {unsqueezed.shape}")

# Permuter les dimensions
x = torch.randn(2, 3, 4)
permuted = x.permute(2, 0, 1)  # (4, 2, 3)
print(f"\npermute(2,0,1): {x.shape} -> {permuted.shape}")

# Transpose (échanger 2 dimensions)
transposed = x.transpose(0, 1)  # Échange dim 0 et 1
print(f"transpose(0,1): {x.shape} -> {transposed.shape}")

# Cas spécial: .T pour matrices 2D
mat = torch.randn(3, 4)
print(f".T (transpose 2D): {mat.shape} -> {mat.T.shape}")

# ============================================================
# 4. INDEXATION ET SLICING
# ============================================================
print("\n" + "="*70)
print("4. INDEXATION ET SLICING")
print("="*70)

x = torch.arange(20).reshape(4, 5)
print(f"Tenseur original:\n{x}")

# Indexation basique
print(f"\nx[0]: {x[0]}")  # Première ligne
print(f"x[:, 0]: {x[:, 0]}")  # Première colonne
print(f"x[1, 3]: {x[1, 3]}")  # Élément (1, 3)

# Slicing
print(f"\nx[1:3]: Lignes 1 et 2:\n{x[1:3]}")
print(f"x[:, 2:4]: Colonnes 2 et 3:\n{x[:, 2:4]}")
print(f"x[::2]: Lignes paires:\n{x[::2]}")

# Indexation avancée
indices = torch.tensor([0, 2, 3])
print(f"\nx[indices]: Lignes 0, 2, 3:\n{x[indices]}")

# Masque booléen
mask = x > 10
print(f"\nMasque (x > 10):\n{mask}")
print(f"x[mask]: {x[mask]}")

# where: condition ternaire
result = torch.where(x > 10, x, torch.zeros_like(x))
print(f"\nwhere(x>10, x, 0): Remplace <10 par 0:\n{result}")

# ============================================================
# 5. OPERATIONS ARITHMETIQUES
# ============================================================
print("\n" + "="*70)
print("5. OPERATIONS ARITHMETIQUES")
print("="*70)

a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(f"a = {a}")
print(f"b = {b}")

# Opérations élément par élément
print(f"\na + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")  # Produit élément par élément
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")

# Versions in-place (modifie le tenseur)
c = a.clone()
c += 10
print(f"\nc += 10: {c}")

# Opérations avec scalaires
print(f"a * 2 = {a * 2}")
print(f"a + 10 = {a + 10}")

# Fonctions mathématiques
x = torch.tensor([0, np.pi/4, np.pi/2])
print(f"\nx = {x}")
print(f"sin(x) = {torch.sin(x)}")
print(f"cos(x) = {torch.cos(x)}")
print(f"exp(x) = {torch.exp(x)}")
print(f"log(x+1) = {torch.log(x + 1)}")
print(f"sqrt(x+1) = {torch.sqrt(x + 1)}")

# Valeur absolue, arrondi, etc.
y = torch.tensor([-2.5, -1.2, 0.5, 1.7, 2.9])
print(f"\ny = {y}")
print(f"abs(y) = {torch.abs(y)}")
print(f"round(y) = {torch.round(y)}")
print(f"floor(y) = {torch.floor(y)}")
print(f"ceil(y) = {torch.ceil(y)}")
print(f"clamp(y, -1, 1) = {torch.clamp(y, -1, 1)}")  # Limite entre -1 et 1

# ============================================================
# 6. OPERATIONS DE REDUCTION
# ============================================================
print("\n" + "="*70)
print("6. OPERATIONS DE REDUCTION")
print("="*70)

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)
print(f"Tenseur:\n{x}")

# Réductions sur tout le tenseur
print(f"\nsum(): {x.sum()}")
print(f"mean(): {x.mean()}")
print(f"std(): {x.std()}")
print(f"max(): {x.max()}")
print(f"min(): {x.min()}")
print(f"prod(): {x.prod()}")  # Produit de tous les éléments

# Réductions par dimension
print(f"\nsum(dim=0): Somme par colonne: {x.sum(dim=0)}")
print(f"sum(dim=1): Somme par ligne: {x.sum(dim=1)}")
print(f"mean(dim=1, keepdim=True): Moyenne par ligne:\n{x.mean(dim=1, keepdim=True)}")

# Argmax/Argmin: indices du max/min
print(f"\nargmax(): Indice du max global: {x.argmax()}")
print(f"argmax(dim=1): Indices max par ligne: {x.argmax(dim=1)}")

# ============================================================
# 7. ALGEBRE LINEAIRE
# ============================================================
print("\n" + "="*70)
print("7. ALGEBRE LINEAIRE")
print("="*70)

# Produit matriciel
A = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6],
                  [7, 8]], dtype=torch.float32)

print(f"A:\n{A}")
print(f"B:\n{B}")

# Plusieurs façons de faire le produit matriciel
print(f"\nA @ B (recommandé):\n{A @ B}")
print(f"torch.mm(A, B):\n{torch.mm(A, B)}")
print(f"torch.matmul(A, B):\n{torch.matmul(A, B)}")

# Produit scalaire (dot product) pour vecteurs
v1 = torch.tensor([1, 2, 3], dtype=torch.float32)
v2 = torch.tensor([4, 5, 6], dtype=torch.float32)
print(f"\nv1 · v2 = {torch.dot(v1, v2)}")

# Batch matrix multiplication
batch1 = torch.randn(3, 2, 4)  # 3 matrices 2x4
batch2 = torch.randn(3, 4, 5)  # 3 matrices 4x5
result = torch.bmm(batch1, batch2)  # 3 matrices 2x5
print(f"\nbmm: {batch1.shape} @ {batch2.shape} -> {result.shape}")

# Transpose
print(f"\nA.T:\n{A.T}")

# Inverse
A_inv = torch.inverse(A)
print(f"\ninverse(A):\n{A_inv}")
print(f"A @ A_inv:\n{A @ A_inv}")

# Déterminant
det = torch.det(A)
print(f"\ndet(A): {det}")

# Valeurs/vecteurs propres
eigenvalues, eigenvectors = torch.linalg.eig(A)
print(f"\nValeurs propres: {eigenvalues}")
print(f"Vecteurs propres:\n{eigenvectors}")

# Norme
v = torch.tensor([3.0, 4.0])
print(f"\nv = {v}")
print(f"Norme L2 (Euclidienne): {torch.norm(v)}")
print(f"Norme L1: {torch.norm(v, p=1)}")

# ============================================================
# 8. CONCATENATION ET EMPILAGE
# ============================================================
print("\n" + "="*70)
print("8. CONCATENATION ET EMPILAGE")
print("="*70)

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

print(f"x:\n{x}")
print(f"y:\n{y}")

# Concatenation
cat_0 = torch.cat([x, y], dim=0)  # Vertical
cat_1 = torch.cat([x, y], dim=1)  # Horizontal
print(f"\ncat dim=0 (vertical):\n{cat_0}")
print(f"cat dim=1 (horizontal):\n{cat_1}")

# Stack: ajoute une nouvelle dimension
stacked_0 = torch.stack([x, y], dim=0)
print(f"\nstack dim=0: {stacked_0.shape}\n{stacked_0}")

# Split: découper en morceaux
chunks = torch.chunk(cat_0, 2, dim=0)
print(f"\nchunk en 2 morceaux: {len(chunks)} tenseurs")
print(f"Chunk 0:\n{chunks[0]}")

# ============================================================
# 9. BROADCASTING
# ============================================================
print("\n" + "="*70)
print("9. BROADCASTING")
print("="*70)

# Broadcasting: opérations entre tenseurs de formes différentes
a = torch.tensor([[1, 2, 3]])  # (1, 3)
b = torch.tensor([[10], [20], [30]])  # (3, 1)

print(f"a {a.shape}:\n{a}")
print(f"b {b.shape}:\n{b}")
print(f"\na + b (broadcasting):\n{a + b}")

# Exemple: normaliser par colonne
data = torch.tensor([[1, 2, 3],
                     [4, 5, 6]], dtype=torch.float32)
col_mean = data.mean(dim=0, keepdim=True)
normalized = data - col_mean
print(f"\nDonnées:\n{data}")
print(f"Moyenne par colonne: {col_mean}")
print(f"Normalisé:\n{normalized}")

# ============================================================
# 10. GRADIENT ET AUTOGRAD
# ============================================================
print("\n" + "="*70)
print("10. GRADIENT ET AUTOGRAD")
print("="*70)

# Créer un tenseur avec requires_grad=True
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Opération
z = x**2 + 2*y + 1
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = x² + 2y + 1 = {z.item()}")

# Calculer les gradients
z.backward()

print(f"\n∂z/∂x = 2x = {x.grad.item()}")
print(f"∂z/∂y = 2 = {y.grad.item()}")

# Désactiver le gradient
with torch.no_grad():
    result = x * 2
print(f"\nAvec no_grad, requires_grad = {result.requires_grad}")

# ============================================================
# 11. CONVERSION DE TYPE
# ============================================================
print("\n" + "="*70)
print("11. CONVERSION DE TYPE")
print("="*70)

x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Original (float32): {x.dtype}")

# Changer le type
x_int = x.to(torch.int64)
x_double = x.double()  # Raccourci pour float64
x_float = x_int.float()  # Raccourci pour float32

print(f"to(int64): {x_int.dtype}")
print(f".double(): {x_double.dtype}")
print(f".float(): {x_float.dtype}")

# Vers/depuis numpy
tensor = torch.tensor([1, 2, 3])
array = tensor.numpy()
back_to_tensor = torch.from_numpy(array)
print(f"\nTenseur -> numpy -> tenseur: {back_to_tensor}")

# Vers Python
value = torch.tensor([42])
python_num = value.item()
python_list = torch.tensor([1, 2, 3]).tolist()
print(f".item(): {python_num} (type: {type(python_num)})")
print(f".tolist(): {python_list} (type: {type(python_list)})")

# ============================================================
# 12. OPERATIONS UTILES
# ============================================================
print("\n" + "="*70)
print("12. OPERATIONS UTILES")
print("="*70)

# Clonage (copie profonde)
x = torch.tensor([1, 2, 3])
y = x.clone()
y[0] = 999
print(f"Original après clone modifié: {x}")

# Répétition
x = torch.tensor([[1, 2]])
repeated = x.repeat(3, 2)  # Répète 3x en dim0, 2x en dim1
print(f"\nrepeat(3, 2):\n{repeated}")

# Unique
x = torch.tensor([1, 2, 2, 3, 3, 3])
unique_vals = torch.unique(x)
print(f"\nOriginal: {x}")
print(f"unique(): {unique_vals}")

# Sort
x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
sorted_vals, sorted_indices = torch.sort(x)
print(f"\nOriginal: {x}")
print(f"sorted: {sorted_vals}")
print(f"indices: {sorted_indices}")

# Topk: k plus grandes valeurs
values, indices = torch.topk(x, k=3)
print(f"\ntop 3 valeurs: {values}")
print(f"leurs indices: {indices}")

# ============================================================
# 13. DEVICE (CPU vs GPU)
# ============================================================
print("\n" + "="*70)
print("13. DEVICE (CPU vs GPU)")
print("="*70)

# Vérifier si GPU disponible
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Créer sur GPU
    x_gpu = torch.tensor([1, 2, 3], device='cuda')
    print(f"Tenseur GPU: {x_gpu.device}")

    # Déplacer de CPU vers GPU
    x_cpu = torch.tensor([1, 2, 3])
    x_gpu = x_cpu.to('cuda')

    # Ramener sur CPU
    x_cpu_back = x_gpu.cpu()
else:
    print("GPU non disponible, utilisation du CPU uniquement")

# Créer automatiquement sur le bon device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device)
print(f"Device utilisé: {device}")

print("\n" + "="*70)
print("FIN DE LA CHEATSHEET")
print("="*70)
print("\nConseil: Bookmarquez ce fichier pour référence rapide!")
