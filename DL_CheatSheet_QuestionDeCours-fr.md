# 🧠 Deep Learning — Ultimate Question de Cours Cheat Sheet

> **Exam-ready. No fluff. Every answer at model level.**

---

## 1. SUPERVISED vs UNSUPERVISED LEARNING

| | Supervisé | Non Supervisé |
|--|-----------|---------------|
| **Data** | Paires (x, y) — avec étiquettes | x seulement — sans étiquettes |
| **Goal** | Apprendre x → y | Apprendre la structure intrinsèque |
| **Examples** | Classification, Régression | Clustering, Autoencoders, GANs |

> **Critical trap:** Supervisé = AVEC étiquettes. Non supervisé = SANS étiquettes.

---

## 2. BACKPROPAGATION

Two phases:

**Phase 1 — Forward Pass:**
- Input x propagates layer by layer → output ŷ computed
- Loss calculated: L = f(ŷ, y)

**Phase 2 — Backward Pass:**
- Compute gradient of loss w.r.t every weight using the **chain rule**:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

- Update weights: **w = w − η · ∂L/∂w**

> **Key phrase to always write:** "Backprop exploite la règle de dérivation en chaîne (chain rule) pour propager le gradient de la sortie vers l'entrée."

---

## 3. OVERFITTING

### Signs:
- Training accuracy >> Validation accuracy
- Training loss ↓ while validation loss ↑ then ↗
- Model **memorizes** training data instead of learning general patterns

### How to avoid (cite at least 3):

| Method | Mechanism |
|--------|-----------|
| **Dropout** | Randomly disables neurons during training |
| **L1 Regularization** | Adds Σ\|wi\| penalty → sparse weights |
| **L2 Regularization** | Adds Σwi² penalty → small weights |
| **Data Augmentation** | Artificially increases dataset size |
| **Early Stopping** | Stop training when val loss starts increasing |
| **Batch Normalization** | Stabilizes training, reduces internal covariate shift |

---

## 4. REGULARIZATION

> "La régularisation modifie la fonction de coût en ajoutant un terme de pénalité sur les poids pour réduire l'overfitting:"

$$L_{total} = L_{original} + \lambda \cdot \Omega(w)$$

| Type | Formula | Effect |
|------|---------|--------|
| **L1 (Lasso)** | Ω(w) = Σ\|wi\| | Sparse weights (many = 0) |
| **L2 (Ridge)** | Ω(w) = Σwi² | Small weights, no sparsity |

- λ = regularization strength hyperparameter

---

## 5. GENERALIZATION ERROR

$$\text{Erreur de généralisation} = \text{Erreur test} - \text{Erreur entraînement}$$

> "C'est l'erreur du modèle sur des données **non vues**. Un écart important entre erreur train et test indique de l'**overfitting**."

---

## 6. CROSS-VALIDATION (K-Fold)

**Process:**
1. Split dataset into **k equal folds**
2. Train k times: each time **k-1 folds for training**, 1 fold for validation
3. Final score = **average of k validation scores**

**When to use:**
- Dataset is small
- No independent test set available
- Need reliable estimate of generalization
- Comparing models or hyperparameters

---

## 7. ACTIVATION FUNCTIONS — WHEN TO USE WHICH

| Problem Type | Output Layer Activation | Why |
|-------------|------------------------|-----|
| **Régression** | Linéaire f(x) = x | Output is unbounded continuous value |
| **Classification binaire** | Sigmoïde σ(x) = 1/(1+e⁻ˣ) | Output ∈ (0,1) → probability of class 1 |
| **Classification multi-classe** | Softmax | Outputs sum to 1 → probability distribution |
| **Hidden layers** | ReLU f(x) = max(0,x) | Avoids vanishing gradient, computationally fast |

### Softmax formula:
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

### Why Softmax and NOT Sigmoid for multi-class:
> "Sigmoid outputs are **independent** and don't sum to 1 — you cannot interpret them as a probability distribution over mutually exclusive classes. Softmax ensures Σ outputs = 1."

---

## 8. LOSS FUNCTIONS

| Problem | Loss Function | Formula |
|---------|--------------|---------|
| Régression | MSE | L = (1/N)Σ(yi - ŷi)² |
| Classification binaire | Binary Cross-Entropy | L = -[y·log(ŷ) + (1-y)·log(1-ŷ)] |
| Classification multi-classe | Categorical Cross-Entropy | L = -Σ yc·log(ŷc) |

---

## 9. to_categorical

> "`to_categorical` convertit une étiquette entière en vecteur **one-hot encodé**."

**Example:**
```
Class 2 out of 4 classes → [0, 0, 1, 0]
Class 0 out of 4 classes → [1, 0, 0, 0]
```

**Why needed:** Softmax outputs a probability vector — target must be in the **same vector format** to compute cross-entropy correctly.

---

## 10. POOLING LAYERS

### Role (always cite 3):
1. **Réduction dimensionnelle** — réduit H×W → moins de calcul
2. **Invariance aux translations** — légère translation de l'objet → même sortie
3. **Réduction de l'overfitting** — moins de paramètres en aval

### Types:

| Type | Operation | Use |
|------|-----------|-----|
| **MaxPooling** | Takes max value in window | Detects feature presence, sharp activations |
| **Average Pooling** | Takes mean of window | Smoother, later layers |
| **Global Average Pooling (GAP)** | Averages entire H×W per channel | Replaces Flatten — output = C only |

### MaxPool reduce by factor 2:
- Pool size = 2×2, **stride = 2**, padding = 0
- Proof: Output = (I - 2)/2 + 1 = **I/2** ✅

---

## 11. PADDING: 'same' vs 'valid'

| | padding='valid' | padding='same' |
|--|----------------|----------------|
| **Padding added** | None (P=0) | P = (F-1)/2 |
| **Output size** | (I-F)/S + 1 < I | = I (when S=1) |
| **Stride constraint** | Any | Must be S=1 |
| **Use case** | Reduce spatial dims | Preserve spatial dims |

**Example** (Input 32×32, F=3×3, S=1):
- valid → (32-3)/1 + 1 = **30×30**
- same → **32×32**

> **Critical:** same padding only preserves dimensions when **stride = 1**. With stride > 1, output is always smaller regardless of padding.

---

## 12. CNN OUTPUT SIZE FORMULA

$$\text{Output} = \left\lfloor \frac{I - F + 2P}{S} \right\rfloor + 1$$

| Symbol | Meaning |
|--------|---------|
| I | Input size (H or W) |
| F | Filter size |
| P | Padding |
| S | Stride |

**Output shape = H_out × W_out × num_filters**

---

## 13. DENSE LAYER (Fully Connected)

$$z = W \cdot x + b$$

| Symbol | Shape |
|--------|-------|
| x (input) | ℝ^{n×1} |
| W (weights) | ℝ^{neurons\_out × neurons\_in} |
| b (bias) | ℝ^{neurons\_out × 1} |
| z (output) | ℝ^{neurons\_out × 1} |

> **Never forget the bias term b.**

---

## 14. MATRIX DIMENSIONS FOR A FULL NETWORK

**Given:** N samples, input dim=d, hidden=h neurons, output=k neurons

| Quantity | Shape | Why |
|----------|-------|-----|
| X (input matrix) | ℝ^{N×d} | N samples, d features |
| W_c (hidden weights) | ℝ^{h×d} | h neurons, each connected to d inputs |
| b_c (hidden bias) | ℝ^{h×1} | one bias per hidden neuron |
| A_c (hidden output) | ℝ^{N×h} | N samples, h activations |
| W_s (output weights) | ℝ^{k×h} | k neurons, each connected to h inputs |
| b_s (output bias) | ℝ^{k×1} | one bias per output neuron |
| A_s (final output) | ℝ^{N×k} | N samples, k class scores |

### Equations (ALWAYS use transpose):

$$A_c = \text{ReLU}(X \cdot W_c^T + b_c) \quad \in \mathbb{R}^{N \times h}$$

$$A_s = \text{Softmax}(A_c \cdot W_s^T + b_s) \quad \in \mathbb{R}^{N \times k}$$

> **Transpose rule:** X ∈ ℝ^{N×d} · W^T ∈ ℝ^{d×h} = ℝ^{N×h} ✅

---

## 15. PERCEPTRON — LIMITATIONS

> "Le perceptron simple ne peut résoudre que des problèmes **linéairement séparables** car sa frontière de décision est un **hyperplan** (droite en 2D). Le problème XOR n'est pas linéairement séparable — aucune droite ne peut séparer les classes {(0,0),(1,1)} des classes {(0,1),(1,0)}."

**Solution:** Multi-layer perceptron (MLP) with hidden layers can solve non-linearly separable problems.

### Perceptron Update Rule:
$$\Delta w_i = \beta \times (y^d - \hat{y}) \times x_i$$
$$w_i \leftarrow w_i + \Delta w_i$$

---

## 16. XOR NETWORK CONSTRUCTION

XOR = (x1 OR x2) AND (NOT(x1) OR NOT(x2))  
     = (x1 OR x2) AND (NAND(x1, x2))

| Neuron | Weights | Bias | Implements |
|--------|---------|------|------------|
| Neuron 1 | w1=1, w2=1 | b=-0.5 | OR |
| Neuron 2 | w1=-1, w2=-1 | b=1.5 | NAND |
| Output | w1=1, w2=1 | b=-1.5 | AND |

**Truth table verification:**

| x1 | x2 | OR | NAND | AND(output) | XOR? |
|----|----|----|------|-------------|------|
| 0 | 0 | 0 | 1 | 0 | ✅ |
| 0 | 1 | 1 | 1 | 1 | ✅ |
| 1 | 0 | 1 | 1 | 1 | ✅ |
| 1 | 1 | 1 | 0 | 0 | ✅ |

---

## 17. STEPS OF A DEEP LEARNING PROJECT

1. **Définition du problème** + collecte et prétraitement des données
2. **Choix de l'architecture** du réseau (couches, neurones, activations)
3. **Choix de la fonction de coût** (MSE, Cross-Entropy...)
4. **Choix de l'algorithme d'optimisation** (SGD, Adam, RMSProp...)
5. **Entraînement** du modèle
6. **Évaluation et validation** (métriques, cross-validation)

---

## 18. DROPOUT

- Randomly sets p% of neuron outputs to **0 during training**
- During inference: **shape unchanged**, weights scaled by (1-p)
- **Shape NEVER changes** — common exam trap
- Purpose: regularization, prevents co-adaptation of neurons

---

## 19. BATCH NORMALIZATION

$$\hat{x} = \frac{x - \mu}{\sigma}, \quad y = \gamma\hat{x} + \beta$$

- μ, σ: mean and std of current batch
- γ, β: **learnable parameters**
- **Shape unchanged**
- Benefits: faster training, acts as regularizer, reduces sensitivity to initialization

---

## 20. OUTPUT NEURONS — HOW MANY?

| Task | # Output Neurons | Activation |
|------|-----------------|------------|
| Binary classification | **1** | Sigmoid |
| Multi-class (C classes) | **C** | Softmax |
| Regression (1 value) | **1** | Linear |
| Regression (k values) | **k** | Linear |

**MNIST example:** 10 digits → **10 neurons** + Softmax  
**House price:** continuous value → **1 neuron** + Linear

---

## ⚡ QUICK REFERENCE — SHAPE CHANGES

| Layer | Shape Changes? | Key Rule |
|-------|---------------|----------|
| Conv2D | ✅ Yes | (I-F+2P)/S+1, depth=filters |
| MaxPool/AvgPool | ✅ Yes (H,W only) | Channels STAY same |
| Flatten | ✅ Yes | H×W×C → single number |
| GAP | ✅ Yes | H×W×C → C only |
| Dense | ✅ Yes | Output = # neurons |
| Dropout | ❌ No | Training only, shape preserved |
| BatchNorm | ❌ No | γ, β learnable |
| Activation (ReLU etc.) | ❌ No | Element-wise operation |

---

---

## 🎯 EXERCICES TYPE — QUESTIONS & RÉPONSES MODÈLES

---

### Q1 — Perceptron et réseau de neurones

> **Considérons un PMC constitué d'une couche d'entrée avec 3 neurones [x1, x2, x3], une couche cachée de 4 neurones avec activation ReLU, et une couche de sortie avec 2 neurones. On dispose de N=100 échantillons d'entraînement.**
>
> **(a)** Donnez la forme de la matrice d'entrée X, la matrice de poids W_c et le vecteur de biais b_c de la couche cachée.
>
> **(b)** Donnez la forme de la matrice de sortie A_c de la couche cachée et écrivez l'équation qui la calcule.
>
> **(c)** Donnez la forme de W_s, b_s et A_s de la couche de sortie. S'il s'agit d'une classification binaire, quelle fonction d'activation utiliser ?

---

**✅ RÉPONSE MODÈLE**

**Rappel de l'architecture :**
- Entrée : d = 3, Couche cachée : h = 4, Sortie : k = 2, N = 100 échantillons

---

**(a) Formes de X, W_c, b_c :**

La matrice d'entrée contient N échantillons, chacun de dimension d=3 :

$$X \in \mathbb{R}^{N \times d} = \mathbb{R}^{100 \times 3}$$

La couche cachée possède h=4 neurones, chacun connecté à d=3 entrées :

$$W_c \in \mathbb{R}^{h \times d} = \mathbb{R}^{4 \times 3}$$

$$b_c \in \mathbb{R}^{h \times 1} = \mathbb{R}^{4 \times 1}$$

---

**(b) Forme de A_c et équation :**

La sortie de la couche cachée contient N activations de dimension h=4 :

$$A_c \in \mathbb{R}^{N \times h} = \mathbb{R}^{100 \times 4}$$

**Équation** (attention à la transposée) :

$$A_c = \text{ReLU}(X \cdot W_c^T + b_c)$$

**Vérification des dimensions :**
$$X \cdot W_c^T = \mathbb{R}^{100 \times 3} \cdot \mathbb{R}^{3 \times 4} = \mathbb{R}^{100 \times 4} \checkmark$$

---

**(c) Formes de W_s, b_s, A_s et fonction d'activation :**

La couche de sortie possède k=2 neurones, chacun connecté à h=4 entrées :

$$W_s \in \mathbb{R}^{k \times h} = \mathbb{R}^{2 \times 4}$$

$$b_s \in \mathbb{R}^{k \times 1} = \mathbb{R}^{2 \times 1}$$

$$A_s \in \mathbb{R}^{N \times k} = \mathbb{R}^{100 \times 2}$$

**Équation :**
$$A_s = f(A_c \cdot W_s^T + b_s)$$

**Vérification :**
$$A_c \cdot W_s^T = \mathbb{R}^{100 \times 4} \cdot \mathbb{R}^{4 \times 2} = \mathbb{R}^{100 \times 2} \checkmark$$

Pour une **classification binaire** avec 2 neurones de sortie → fonction **Softmax**, car les deux classes sont mutuellement exclusives et les sorties doivent sommer à 1.

> ⚠️ Si la couche de sortie n'avait qu'**1 neurone** (aussi valide pour binaire) → **Sigmoïde**, car la sortie représente P(classe=1) ∈ (0,1).

---

### Q2 — CNN : calcul des dimensions

> **Soit un CNN avec l'architecture suivante appliquée à une image RGB de taille 64×64×3 :**
> - **(a)** Conv2D : 32 filtres, 3×3, stride=1, padding='valid'
> - **(b)** MaxPooling : 2×2, stride=2
> - **(c)** Conv2D : 64 filtres, 3×3, stride=1, padding='same'
> - **(d)** Global Average Pooling
> - **(e)** Dense(10) — classification multi-classe
>
> **Calculez la taille de la sortie après chaque couche.**

---

**✅ RÉPONSE MODÈLE**

**Formule :** Output = ⌊(I − F + 2P) / S⌋ + 1

| Couche | Calcul | Sortie |
|--------|--------|--------|
| Entrée | — | **64 × 64 × 3** |
| (a) Conv2D(32, 3×3, S=1, P=0) | ⌊(64−3+0)/1⌋+1 = 62 | **62 × 62 × 32** |
| (b) MaxPool(2×2, S=2) | ⌊(62−2)/2⌋+1 = 31 | **31 × 31 × 32** |
| (c) Conv2D(64, 3×3, S=1, same) | P=(3−1)/2=1 → sortie=entrée | **31 × 31 × 64** |
| (d) GAP | Moyenne H×W par canal → C seul | **64** |
| (e) Dense(10) | W ∈ ℝ^{10×64}, z=Wx+b | **10** |

**Fonction d'activation de la couche Dense(10) :** **Softmax**, car classification multi-classe à 10 classes → les sorties doivent former une distribution de probabilité.

**Fonction de perte associée :** **Entropie croisée catégorielle** :
$$L = -\sum_{c=1}^{10} y_c \cdot \log(\hat{y}_c)$$

---

### Q3 — Overfitting et régularisation

> **Un modèle de deep learning atteint 98% de précision sur les données d'entraînement mais seulement 61% sur les données de test.**
>
> **(a)** Quel phénomène cela illustre-t-il ? Justifiez.
>
> **(b)** Proposez trois méthodes pour y remédier et expliquez brièvement le mécanisme de chacune.

---

**✅ RÉPONSE MODÈLE**

**(a) Phénomène :**

Il s'agit de **sur-apprentissage (overfitting)**. Le modèle a **mémorisé** les données d'entraînement (bruit inclus) au lieu d'apprendre les patterns généraux. Cela se manifeste par un écart important entre l'erreur d'entraînement (2%) et l'erreur de test (39%) — c'est précisément la définition de l'erreur de généralisation élevée.

**(b) Trois méthodes :**

**1. Dropout (taux p=0.5) :**
À chaque itération d'entraînement, chaque neurone est désactivé aléatoirement avec probabilité p. Cela force le réseau à ne pas dépendre d'un sous-ensemble de neurones et à apprendre des représentations plus robustes.

**2. Régularisation L2 :**
On ajoute une pénalité sur les poids à la fonction de coût :
$$L_{total} = L_{original} + \lambda \sum_i w_i^2$$
Cela décourage les poids trop grands, réduisant la complexité effective du modèle.

**3. Data Augmentation :**
On augmente artificiellement la taille du dataset en appliquant des transformations aux données existantes (rotation, zoom, flip...). Le modèle voit plus de variabilité et généralise mieux.

---

*Last updated for Deep Learning exam — master these 20 topics + 3 exercices types and you cover 95% of all question de cours scenarios.*
