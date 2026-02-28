# 📚 Résumé Complet — Perceptron Multi-Couches (PMC)

---

## 2.1 Limites du Perceptron Simple

### Le perceptron simple et la linéarité
Le perceptron simple est un modèle de base en **classification supervisée binaire**. Il cherche à séparer deux classes à l'aide d'une frontière **linéaire** (droite en 2D, hyperplan en nD).

**Condition de fonctionnement :** les données doivent être **linéairement séparables**, c'est-à-dire qu'il existe une droite (ou hyperplan) capable de séparer correctement les deux classes.

Exemples de fonctions linéairement séparables : **OR**, **AND**.

### La limite fondamentale : le problème XOR
En **1969, Minsky et Papert** ont démontré formellement que le perceptron simple **ne peut pas résoudre** la fonction binaire XOR(x₁, x₂).

**Table de vérité XOR :**

| x₁ | x₂ | XOR |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

**Preuve mathématique que XOR n'est pas linéairement représentable :**

Supposons qu'il existe des paramètres (w₁, w₂, w₀) tels que :

```
XOR(x₁, x₂) = w₁x₁ + w₂x₂ + w₀
```

On teste sur les 4 couples :
- (0, 0) → w₀ = 0
- (1, 0) → w₁ + w₀ = 1
- (0, 1) → w₂ + w₀ = 1
- (1, 1) → w₁ + w₂ + w₀ = 0

En additionnant les 3 premières équations : **w₁ + w₂ + w₀ = 2**

Mais la 4ème impose : **w₁ + w₂ + w₀ = 0**

→ **Contradiction** : XOR ne peut pas être représenté par un modèle linéaire. ✗

---

## 2.1.1 Comment Dépasser ces Limites

**Solution :** Assembler plusieurs perceptrons en **couches successives** → le **Perceptron Multi-Couches (PMC)**.

Grâce à une couche cachée, le PMC peut créer des **transformations non linéaires** des données, ce qui lui permet de résoudre des problèmes impossibles pour un perceptron simple.

Le PMC peut représenter XOR comme une combinaison de portes logiques :
```
x₁ XOR x₂ = (x₁ OR x₂) AND (NOT(x₁) OR NOT(x₂))
```

### Théorème d'Approximation Universelle
Avec suffisamment de couches et de neurones, un PMC peut **approximer n'importe quelle fonction** de classification ou de régression. C'est le **théorème d'approximation universelle**.

**Synthèse :**
- Perceptron simple → uniquement problèmes **linéaires**
- PMC → problèmes **complexes et non linéaires**

---

## 2.2 Structure d'un PMC

Un PMC est composé de **3 types de couches** :

### Couche d'entrée
- Reçoit les variables d'entrée (caractéristiques des données)
- Nombre de neurones = **nombre de features** dans les données

### Couches cachées
- Une ou plusieurs couches **intermédiaires**
- Permettent de modéliser des relations complexes
- Nombre de neurones choisi **par expérimentation**
- Plus il y a de neurones → plus le réseau est puissant, mais plus le risque de **surapprentissage** augmente
- Fonctions d'activation généralement **identiques** dans toutes les couches cachées

### Couche de sortie
- Produit la **sortie finale** du réseau
- Nombre de neurones dépend de la tâche :
  - Régression simple → 1 neurone
  - Régression multivariée → autant de neurones que de valeurs à prédire
  - Classification binaire → 1 neurone (sigmoïde)
  - Classification multiclasse → 1 neurone **par classe** (softmax)

### Connexions
- Les neurones d'une **même couche ne sont PAS connectés** entre eux
- Chaque neurone est connecté à **TOUS** les neurones de la couche précédente et suivante
- Architecture dite **fully connected** ou **dense**

### Réseau de neurones profond
Quand un réseau possède **au moins 2 couches cachées** → on parle de **réseau de neurones profond (Deep Neural Network)**

---

## 2.2.1 Choix de la Structure du Réseau

| Couche | Critère de choix |
|--------|-----------------|
| Entrée | Nombre de features (fixé par les données) |
| Cachées | Par expérimentation (trial & error) |
| Sortie | Dépend de la tâche |

**Couche de sortie selon la tâche :**

| Tâche | Nb neurones sortie | Activation |
|-------|--------------------|-----------|
| Régression simple | 1 | Linéaire (aucune) |
| Régression multivariée | p (nb valeurs) | Linéaire |
| Classification binaire | 1 | Sigmoïde |
| Classification multiclasse (K classes) | K | Softmax |

---

## 2.3 Sortie d'un PMC

### 2.3.1 Régression — Formule de sortie

La couche de sortie **n'utilise pas de fonction d'activation**. La sortie du neurone j est :

$$\hat{y}_j = \sum_{i=1}^{R} w(i,j)\hat{y}_i + w(0,j)$$

où :
- w(i,j) = poids connectant le neurone i de la couche précédente au neurone j
- w(0,j) = biais du neurone j
- ŷᵢ = sortie du neurone i connecté à j
- R = nombre de neurones de la couche précédente

### 2.3.2 Classification Binaire

- Étiquettes : yᵈ ∈ {0, 1} → classes C₁ et C₂
- 1 seul neurone en sortie avec **fonction sigmoïde** σ :

$$\hat{y}_j = \sigma\left(\sum_{i=1}^{R} w(i,j)\hat{y}_i + w(0,j)\right)$$

**Règle de décision :**
- Si ŷⱼ > 0.5 → classe **C₁**
- Sinon → classe **C₂**

### 2.3.3 Classification Multiclasse

- K classes, étiquettes : Y = (y₁ᵈ, ..., yₖᵈ) ∈ ℝᴷ
- K neurones en sortie avec **fonction softmax** :

$$\hat{y}_j = \text{softmax}\left(\sum_{i=1}^{R} w(i,j)\hat{y}_i + w(0,j)\right)$$

- Classe prédite : ŷᵢ = arg max(ŷⱼ)

**Encodage One-Hot :** la classe prédite prend la valeur 1, les autres 0.
- Exemple CIFAR-10 : "airplane" → [1,0,0,0,0,0,0,0,0,0]

---

## 2.4 Apprentissage et Rétropropagation des Erreurs

### 2.4.1 Mesure d'erreur d'un neurone de sortie (Gradient local)

Pour le neurone j de la couche de sortie [k] :

$$\delta_j^{[k]} = \hat{y}_j^{[k]}(1 - \hat{y}_j^{[k]})\left(y_j^{d[k]} - \hat{y}_j^{[k]}\right)$$

**Étapes de calcul :**
1. **Erreur brute :** eⱼ = yⱼᵈ - ŷⱼ
2. **Dérivée de la sigmoïde :** σ'(x) = ŷⱼ(1 - ŷⱼ)
3. **Gradient local :** δⱼ = σ'(xⱼ) · eⱼ

Le gradient local sert à :
- Mettre à jour les poids lors de la rétropropagation
- Propager l'erreur vers la couche précédente

### 2.4.2 Mise à jour des poids — Couche de sortie

**Gradient global :**
$$\delta^{[k]} = \hat{Y}^{[k]}(1 - \hat{Y}^{[k]})(Y^{d[k]} - \hat{Y}^{[k]})$$

**Mise à jour des poids :**
$$\Delta W^{[k]} = \frac{\beta}{m} \delta^{[k]} (\hat{Y}^{[k-1]})^T$$

**Mise à jour des biais :**
$$\Delta b^{[k]} = \frac{\beta}{m} \sum_{j=1}^{m} \delta_j^{[k]}$$

où β est le taux d'apprentissage et m le nombre d'exemples.

### 2.4.3 Mise à jour des poids — Couche cachée

L'erreur des neurones cachés est calculée à partir de la couche suivante :

$$\delta_j^{[k-1]} = \hat{y}_j^{[k-1]}(1 - \hat{y}_j^{[k-1]}) \sum_{r \in \text{dest}(j)} w^{[k]}(j, r) \delta_r^{[k]}$$

En notation vectorielle :
$$\delta^{[k-1]} = \hat{Y}^{[k-1]} \odot (1 - \hat{Y}^{[k-1]}) \cdot (W^{[k]T} \delta^{[k]})$$

où ⊙ est la multiplication élément par élément (Hadamard).

**Mise à jour des poids :**
$$\Delta W^{[k-1]} = \frac{\beta}{m} \delta^{[k-1]} (\hat{Y}^{[k-2]})^T$$

**Mise à jour des biais :**
$$\Delta b^{[k-1]} = \frac{\beta}{m} \sum_{j=1}^{m} \delta_j^{[k-1]}$$

### 2.4.4 Rétropropagation du Gradient d'Erreur (Backpropagation)

La rétropropagation consiste à :
1. Calculer les δⱼ pour les **neurones de sortie**
2. Propager progressivement ces erreurs **vers les couches cachées**, jusqu'aux neurones d'entrée
3. **Modifier les poids** des connexions pour réduire l'écart entre sorties prédites et valeurs réelles

Direction : sortie → entrée (sens inverse de la propagation avant)

### 2.4.5 Phénomène de Saturation des Neurones

La sortie d'un neurone j avec sigmoïde :
$$\hat{y}_j = \text{sigmoïde}(y), \quad y = \sum_{i=1}^{n} x_i w_i + w_0$$

Un neurone est dit **saturé** si :
- ŷⱼ → 0 ⟹ y → -∞
- ŷⱼ → 1 ⟹ y → +∞

Pour des valeurs extrêmes (y < -10 ou y > 10), le gradient est **pratiquement nul** → apprentissage très lent.

**Solution : normaliser les données à l'entrée du PMC.**
- Transformer chaque variable xᵢ dans l'intervalle [-1/max(xᵢ), 1/max(xᵢ)]

### 2.4.6 Valeurs Désirées en Sortie

En classification, les valeurs désirées yᵈ ∈ {0, 1} peuvent causer une **saturation** de la sigmoïde.

**Solution :** Transformer les valeurs désirées en yᵈ ∈ **{0.05, 0.95}**
- Si xᵢ ∈ Cᵢ → yᵢᵈ = **0.95**
- Sinon → yᵢᵈ = **0.05**

### 2.4.7 Initialisation des Poids et Taux d'Apprentissage

#### Initialisation des poids
- Les poids et biais sont initialisés **aléatoirement** avant l'apprentissage
- Distribution uniforme dans **[-0.5, 0.5]**
- Le PMC est donc un **algorithme stochastique** : deux entraînements peuvent donner des résultats légèrement différents

#### Taux d'apprentissage β
- Contrôle l'amplitude des mises à jour des poids
- Valeur généralement entre **0 et 1**
- Choix **empirique** (pas de règle universelle)

**Stratégies :**
- **Diminution progressive :** commencer grand β puis réduire
- **Alternance :** alterner petite/grande valeur de β

### 2.4.8 Déroulement de l'Algorithme d'Apprentissage

1. **Normaliser les données :** xᵢ ∈ [-1, 1], yᵈ ∈ {0.05, 0.95}
2. **Initialiser les poids et biais :** w(i,j) ∈ [-0.5, 0.5]
3. **Répéter** jusqu'au critère d'arrêt :
   a. Choisir aléatoirement un exemple (Xᵢ, yᵢᵈ)
   b. Pour chaque couche (en partant de la sortie) :
      - Calculer δⱼ et Δw(i,j) pour les neurones de sortie
      - Calculer δⱼ et Δw(i,j) pour les neurones cachés
   c. Mettre à jour :
      - w(i,j) ← w(i,j) + Δw(i,j)
      - w(0,j) ← w(0,j) + Δw(0,j)

### 2.4.9 Exemple Complet : Apprentissage XOR

**Paramètres :** sigmoïde, β = 1

**Poids initiaux donnés :**
- w(0,3)=0.2, w(1,3)=0.1, w(2,3)=0.3
- w(0,4)=-0.3, w(1,4)=-0.2, w(2,4)=0.4
- w(0,5)=0.4, w(3,5)=0.5, w(4,5)=-0.4

**Propagation avant pour (x₁,x₂) = (1,1) :**
- S₃ = 0.2 + 0.1×1 + 0.3×1 = 0.6 → ŷ₃ = 0.65
- S₄ = -0.3 + (-0.2)×1 + 0.4×1 = -0.1 → ŷ₄ = 0.48
- S₅ = 0.4 + 0.5×0.65 + (-0.4)×0.48 = 0.53 → ŷ₅ = 0.63

**Rétropropagation :**
- δ₅ = (0 - 0.63) × 0.63 × (1 - 0.63) ≈ **-0.147**
- Δw(3,5) ≈ -0.1, Δw(4,5) ≈ -0.07, Δw(0,5) = -0.147
- δ₄ = 0.48 × (1-0.48) × (-0.147) × (-0.4) ≈ **0.015**
- δ₃ = 0.65 × (1-0.65) × (-0.147) × 0.5 ≈ **-0.017**

**Après mise à jour**, la sortie passe de **0.63 à 0.56** → réduction de l'erreur ✓

---

## 2.5 Sur-apprentissage (Overfitting)

### Concept de généralisation
La **généralisation** est la capacité d'un modèle à obtenir de bons résultats sur des **données non vues** lors de l'entraînement.

| Concept | Définition |
|---------|-----------|
| Erreur d'apprentissage | Calculée sur les données d'entraînement |
| Erreur de généralisation | Calculée sur des données nouvelles |

### Cas possibles
- **Sous-apprentissage (underfitting) :** erreur d'entraînement élevée → le modèle n'apprend pas
- **Surapprentissage (overfitting) :** faible erreur d'entraînement mais forte erreur de test → le modèle **mémorise** au lieu d'apprendre

---

## 2.6 Régularisation

La régularisation regroupe les méthodes visant à **réduire le surapprentissage** et améliorer la **généralisation**.

### Trois axes principaux :
1. **Modifier la fonction d'erreur** (pénalisation L1/L2)
2. **Modifier les données** (augmentation de données)
3. **Modifier le modèle** (dropout)

### 2.6.1 Modification de la Fonction d'Erreur

On ajoute un terme de pénalisation Ω(w) :
$$\widetilde{Err}_i(w) = Err_i(w) + \Omega(w)$$

**L1 — Lasso :**
$$\Omega(w) = \lambda \sum_{i=1}^{N} |w_i|$$
→ Favorise des poids **nuls** (sélection de features)

**L2 — Ridge :**
$$\Omega(w) = \lambda \sum_{i=1}^{N} w_i^2$$
→ Favorise des poids **petits**

**Rôle de λ :**
- λ = 0 → pas de régularisation
- λ grand → régularisation forte, réduit le surapprentissage

### 2.6.2 Augmentation de Données

Créer artificiellement de nouvelles données pour améliorer la généralisation.

**Méthodes pour les images :**
- Translation
- Rotation
- Changement d'échelle
- Symétrie (horizontal/vertical flip)
- Random crop

### 2.6.3 Dropout

Le **dropout** désactive temporairement un pourcentage de neurones (ex: 50%) **pendant l'entraînement**.

**Objectif :** améliorer la généralisation et éviter la dépendance à un petit nombre de neurones.

**Application :** souvent dans les couches entièrement connectées vers la fin du réseau.

**Effet :** les courbes train/test se rapprochent → meilleure généralisation.
