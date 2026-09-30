# PyTorch Grind

Repository pour suivre mon apprentissage et ma pratique de PyTorch.

## Contenu

### Fondamentaux
- `gradient_descent_visualization.py` - Visualisation de la descente de gradient
  - Comparaison calcul manuel vs `loss.backward()`
  - Surface de loss en 3D et trajectoire
  - Évolution des paramètres et gradients
  - 5 graphiques détaillés + logs TensorBoard

### Régression Linéaire
- `linear_regression_graph.py` - Visualisation du graphe de calcul avec torchviz
- `tensorboard_linear_regression.py` - Régression linéaire avec logs TensorBoard

### Classification d'Images
- `tensorboard_image_classification.py` - CNN complet sur MNIST avec visualisations avancées
- `tensorboard_model_comparison.py` - Comparaison de 5 architectures de complexité croissante
  - Modèle simple linéaire (91.57%)
  - Une couche cachée (96.58%)
  - Deux couches cachées (97.22%)
  - Avec dropout (96.28%)
  - CNN (99.10%)

## Utilisation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer TensorBoard
tensorboard --logdir=runs

# Exécuter les scripts
python tensorboard_model_comparison.py
```

## Progression

Le dossier `runs/` contient les logs TensorBoard pour visualiser:
- Courbes de loss et accuracy
- Architecture des modèles
- Distribution des poids
- Embeddings et visualisations

---

*Repository de grind PyTorch - Learning by doing*
