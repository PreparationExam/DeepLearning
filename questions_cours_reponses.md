# Questions de Cours — Réponses Complètes
> Deep Learning — PMC, CNN, Régularisation

---

## BLOC 1 — Fonctions d'Activation (Couche de Sortie)

### Q : Pour un problème de **régression**, quelle fonction d'activation utiliser en sortie ?
**Identité (linéaire) : f(z) = z**
La sortie doit être non bornée (valeur continue). Appliquer sigmoid/softmax détruirait la prédiction en écrasant la sortie dans [0,1].

---

### Q : Pour un problème de **classification bi-classe**, quelle fonction d'activation utiliser en sortie ?
**Sigmoid : σ(z) = 1/(1+e⁻ᶻ)**
Sortie ∈ [0,1] interprétée comme P(classe=1). Un seul neurone de sortie suffit.

---

### Q : Pour un problème de **classification multi-classe**, quelle fonction d'activation utiliser en sortie ?
**Softmax :**

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Les sorties somment à 1 → distribution de probabilité sur K classes. Le dénominateur normalise sur **tous les neurones de la même couche** (j ne désigne pas la couche précédente).

---

## BLOC 2 — Formes des Matrices (Architecture PMC)

> Contexte : entrée 10 features, couche cachée 50 neurones, sortie 3 neurones, N échantillons.

### Q : Quelle est la forme de la matrice d'entrée X ?
$$X \in \mathbb{R}^{N \times 10}$$
N lignes (échantillons), 10 colonnes (features).

---

### Q : Quelle est la forme de Wc (poids couche cachée) et bc (biais couche cachée) ?
$$W_c \in \mathbb{R}^{10 \times 50}, \quad b_c \in \mathbb{R}^{1 \times 50}$$
Règle : W a la forme **(entrée × sortie)**. Biais = 1 valeur par neurone.

---

### Q : Quelle est la forme de la matrice de sortie Ac (couche cachée) ?
$$A_c \in \mathbb{R}^{N \times 50}$$
N échantillons, 50 activations par échantillon.

---

### Q : Écrire l'équation qui calcule Ac en fonction de X, Wc et bc.
$$A_c = \text{ReLU}(X \cdot W_c + b_c)$$
Vérification dimensions : (N×10)(10×50) = (N×50) ✅. ReLU appliqué élément par élément : ReLU(z) = max(0, z).

---

### Q : Quelle est la forme de Ws (poids couche de sortie) et bs ?
$$W_s \in \mathbb{R}^{50 \times 3}, \quad b_s \in \mathbb{R}^{1 \times 3}$$

---

### Q : Quelle est la forme de la matrice de sortie As du réseau ?
$$A_s \in \mathbb{R}^{N \times 3}$$

---

### Q : Écrire l'équation de As (classification multi-classe) en fonction de Ac, Ws et bs.
$$A_s = \text{Softmax}(A_c \cdot W_s + b_s)$$
Vérification : (N×50)(50×3) = (N×3) ✅. Softmax appliqué **ligne par ligne** (par échantillon).

---

## BLOC 3 — Perceptron Simple

### Q : Décrire un problème que le perceptron simple ne peut pas résoudre. Justifier.
**Le problème XOR.**
Le perceptron simple ne peut classifier que des données **linéairement séparables** (sa frontière de décision est un hyperplan). XOR n'est pas linéairement séparable : aucune droite ne peut séparer les classes (0,0),(1,1) des classes (0,1),(1,0). Il faut au minimum une couche cachée (MLP) pour résoudre XOR.

---

### Q : Dessiner un réseau implémentant x1 XOR x2, sachant que x1 XOR x2 = (x1 OR x2) AND (NOT(x1) OR NOT(x2)).

```
x1 ──┬──→ [N1: OR]  ──────────→ [N3: AND] → sortie
     │                                ↑
x2 ──┴──→ [N2: NAND] ─────────────────┘
```

- N1 implémente (x1 OR x2)
- N2 implémente NOT(x1) OR NOT(x2) = NAND(x1,x2)
- N3 implémente AND des deux → résultat XOR

---

## BLOC 4 — MNIST & Classification

### Q : Combien de neurones dans la couche de sortie pour MNIST ? Quelle fonction d'activation ?
**10 neurones** (un par chiffre 0→9) + **Softmax**.
Sortie = distribution de probabilité sur 10 classes mutuellement exclusives.

---

### Q : Pour prédire le prix des maisons (régression), combien de neurones en sortie ? Quelle activation ?
**1 neurone** + **fonction identité (linéaire)**.
La sortie est une valeur continue non bornée → pas de fonction d'écrasement.

---

## BLOC 5 — Généralisation & Overfitting

### Q : Qu'est-ce que l'erreur de généralisation ?
L'erreur de généralisation est la différence entre l'erreur sur les **données de test** (non vues pendant l'entraînement) et l'erreur sur les **données d'entraînement**. Elle mesure la capacité du modèle à performer sur de nouvelles données.

$$E_{gen} = E_{test} - E_{train}$$

---

### Q : La courbe montre que l'erreur de test remonte alors que l'erreur d'entraînement continue de baisser. Quel est ce phénomène ? Comment l'éviter (3 méthodes) ?

**Phénomène : Surapprentissage (Overfitting)**
Le modèle mémorise les données d'entraînement au lieu d'apprendre les patterns généraux. Il performe bien sur le train mais mal sur le test.

**3 méthodes pour l'éviter :**
1. **Régularisation** (L1/L2) : pénalise les poids trop grands pour limiter la complexité du modèle.
2. **Dropout** : désactive aléatoirement des neurones pendant l'entraînement, forçant le réseau à ne pas dépendre d'un seul chemin.
3. **Early stopping** : arrêter l'entraînement quand l'erreur de validation commence à remonter.

---

### Q : Qu'est-ce que la régularisation de manière générale ?
La régularisation est une technique qui **ajoute une pénalité à la fonction de coût** pour contraindre les poids du modèle et réduire le surapprentissage.

$$J_{reg} = J + \lambda \cdot \Omega(W)$$

- **L2 (Ridge)** : Ω(W) = Σwᵢ² → pousse les poids vers 0 sans les annuler
- **L1 (Lasso)** : Ω(W) = Σ|wᵢ| → peut annuler complètement certains poids (sélection de features)
- λ contrôle l'intensité de la régularisation

---

### Q : Quand utiliser la validation croisée ?
La validation croisée (k-fold) est utilisée quand :
- Le **dataset est petit** (pas assez de données pour un split fixe train/val/test)
- On veut une **estimation robuste** des performances du modèle
- On doit **comparer plusieurs modèles** ou hyperparamètres de façon fiable

Principe : diviser les données en k folds, entraîner k fois en changeant le fold de validation, moyenner les performances.

---

## BLOC 6 — CNN : Nombre de Paramètres

### Q : Comment calculer le nombre de paramètres d'une couche Conv ?
$$\text{Paramètres} = (F \times F \times C_{in} + 1) \times C_{out}$$

**Exemple** : Conv 3×3, entrée RGB (3 canaux), 32 filtres :
- Poids = 3×3×3×32 = 864
- Biais = 32
- **Total = 896**

---

### Q : Combien de paramètres pour une couche MaxPool ?
**0 paramètre.** MaxPool est une opération mathématique pure (prendre le maximum). Aucun poids, aucun biais appris.

---

### Q : Comment MaxPool réduit-il par un facteur 2 ?
Avec **pool size = 2×2, stride = 2, padding = 0** :

$$\text{Output} = \frac{I - F}{S} + 1 = \frac{I - 2}{2} + 1 = \frac{I}{2}$$

Le stride = taille du filtre → fenêtre sans chevauchement → réduction exacte par 2.

---

## BLOC 7 — Récapitulatif Activations

| Tâche | Activation sortie | Nb neurones sortie |
|-------|------------------|-------------------|
| Régression | Identité f(z)=z | 1 |
| Classification binaire | Sigmoid | 1 |
| Classification multi-classe | Softmax | K (nb classes) |
| Multi-label | Sigmoid | K (indépendants) |

---

## BLOC 8 — Récapitulatif Formes (règle universelle)

| Élément | Forme |
|---------|-------|
| X (entrée) | (N × d_in) |
| W (poids) | (d_in × d_out) |
| b (biais) | (1 × d_out) |
| A (sortie couche) | (N × d_out) |

**Équation forward pass universelle :**
$$A^{[l]} = f(A^{[l-1]} \cdot W^{[l]} + b^{[l]})$$
