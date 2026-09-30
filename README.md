# PyTorch Grind

Repository pour suivre mon apprentissage et ma pratique de PyTorch.

## Contenu

### Régression Linéaire
- `linear_regression_graph.py` - Visualisation du graphe de calcul avec torchviz
- `tensorboard_linear_regression.py` - Régression linéaire avec logs TensorBoard

### Classification d'Images
- `tensorboard_image_classification.py` - CNN complet sur MNIST avec visualisations avancées
- `tensorboard_model_comparison.py` - Comparaison de 5 architectures de complexité croissante

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
