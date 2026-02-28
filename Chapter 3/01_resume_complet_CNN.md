# 📚 Résumé Complet — Réseaux de Neurones Convolutifs (CNN/RNC)

---

## 1. Convolutions

### 1.1 Convolution d'image

La convolution est une **opération linéaire** entre une image et un filtre (appelé aussi **kernel** ou noyau), qui produit une nouvelle image.

**Formule (image 2D) :**
$$\hat{I}(i,j) = I(i,j) * K(i,j) = \sum_{n=0}^{F-1}\sum_{m=0}^{F-1} K(n,m) \cdot I(i+n, j+m)$$

- La sortie Î est appelée **carte de caractéristiques (feature map)**
- Chaque pixel de la carte = somme pondérée des poids du filtre × pixels voisins correspondants

**Formule (image multi-canaux, C canaux) :**
$$\hat{I}(i,j) = \sum_{k=1}^{C}\sum_{n=0}^{F-1}\sum_{m=0}^{F-1} K(i,j,k) \cdot I(i+n, j+m, k)$$

**Point de vue d'un neurone :** chaque neurone reçoit un patch de l'image et applique un filtre → produit scalaire + biais → activation.

---

## 1.2 Paramètres du Filtre

### 1.2.1 Dimensions d'un filtre

- Filtre de taille **F×F** sur une image de **C canaux** → volume **F×F×C**
- Image d'entrée : **N×M×C**
- Carte de sortie : **H×L×1** (par filtre)

### 1.2.2 Stride (Pas de convolution)

La **stride S** = nombre de pixels dont la fenêtre se déplace après chaque opération.

- S = 1 ou 2 généralement, rarement plus
- **Taille de sortie avec stride S :**
$$L = \frac{M - F}{S} + 1 \quad \text{et} \quad H = \frac{N - F}{S} + 1$$

**⚠️ Problème :** Convolution impossible pour les **pixels en bordure** → solution : Padding

**Exemples :**
- Entrée 7×7, filtre 3×3, S=1 → sortie **5×5**
- Entrée 7×7, filtre 3×3, S=2 → sortie **3×3**
- Entrée 7×7, filtre 3×3, S=3 → **impossible** (pas de nombre entier)

### 1.2.3 Padding (Remplissage)

**Problème :** Convolution impossible pour les pixels en bordure (pas de voisins suffisants).

**Solution :** Agrandir l'image avant convolution.

**Méthodes :**

1. **Zero-padding :** Ajouter P zéros à chaque côté des frontières
   - Filtre F×F → padding P = (F-1)/2
   - Filtre 3×3 → P = 1
   - Filtre 5×5 → P = 2
   - Filtre 7×7 → P = 3

2. **Duplication :** Dupliquer les premières/dernières lignes et colonnes au-delà des bords

**Taille de sortie avec padding P et stride S :**
$$L = \frac{M - F + 2P}{S} + 1 \quad \text{et} \quad H = \frac{N - F + 2P}{S} + 1$$

**Exemples :**
- Image 7×7, filtre 3×3, P=1 → image paddée 9×9 → sortie **7×7** (même taille !)
- Image 7×7, filtre 5×5, P=2 → image paddée 11×11 → sortie **7×7**

### 1.2.4 Exemples de Filtres Classiques

| Filtre | Matrice | Effet |
|--------|---------|-------|
| **Identité** | [[0,0,0],[0,1,0],[0,0,0]] | Image inchangée |
| **Détection de contours** | [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] | Détecte les bords |
| **Amélioration netteté** | [[0,-1,0],[-1,5,-1],[0,-1,0]] | Renforce les détails |

---

## 2. Réseaux de Neurones Convolutionnels

### 2.1 Pourquoi des CNN ?

**Problème avec les PMC pour les images :**
- Une image 224×224×3 = 150 528 pixels → millions de poids si fully connected → **impossible à calculer**

**Idée des CNN :**
- Comparer les images **fragment par fragment** (pas l'image entière)
- **Partage de poids :** les neurones d'une même couche partagent le même filtre → appliqué sur toute l'image
- **Extraction hiérarchique :**
  - Premières couches → caractéristiques **bas niveau** (bords, textures) locales
  - Dernières couches → formes **haut niveau** (objets, visages) globales
- Seule la dernière couche exploite les informations de sortie

### 2.2 Architecture d'un CNN

Architecture typique :
```
[Image] → [Conv + ReLU] → [Pooling] → [Conv + ReLU] → [Pooling] → [FC] → [Softmax] → [Classes]
```

#### 2.2.1 Couche de Convolution

- Utilise des filtres qui **scannent l'entrée** en effectuant des convolutions
- Réglable via : taille du filtre F, stride S, padding P
- Appliquer **N filtres** de taille F×F engendre une carte de taille **H×L×N**

**Banque de filtres :** N filtres → N cartes de caractéristiques empilées en profondeur

#### 2.2.2 Fonctions d'Activation

Appliquée entre les couches de convolution et de pooling.

| Fonction | Formule | Avantage |
|----------|---------|---------|
| **ReLU** | g(z) = max(0, z) | Complexités non-linéaires, interprétable biologiquement |
| **Leaky ReLU** | g(z) = max(εz, z), ε ≪ 1 | Résout le problème "dying ReLU" |
| **ELU** | g(z) = max(α(eˣ−1), z), α ≪ 1 | Dérivable partout |

**ReLU** est de loin la plus utilisée dans les CNN.

#### 2.2.3 Couche de Pooling (Sous-échantillonnage)

La sortie de convolution est découpée en **zones sans chevauchement** et réduite :

| Type | Opération | Résultat |
|------|-----------|---------|
| **Max Pooling** | Prend la valeur maximale | Conserve les features les plus saillantes |
| **Average/Mean Pooling** | Prend la valeur moyenne | Lissage, moins agressif |

**Effets :**
- Réduction de la taille des cartes
- Augmente l'**invariance** aux transformations (rotation, translation, symétrie, échelle)

**Exemple :** Max pooling filtre 2×2, stride=2 → divise la taille par 2

#### 2.2.4 Couche Totalement Connectée (FC)

- Chaque entrée connectée à **tous les neurones** (comme dans un PMC)
- Typiquement à la **fin** des CNN
- Sert à **classifier** (calculer les scores de classe)
- Cas particulier : si la taille du filtre = taille de l'entrée → équivalent à une couche FC

---

## 3. Évolution des Architectures CNN (ImageNet)

### 3.1 ImageNet & ILSVRC

- **Dataset :** 15 millions d'images, 22 000 catégories
- **ILSVRC** (depuis 2010) : 1 000 catégories, 1 000 images/catégorie, 50 000 val, 150 000 test
- Objectif : faire avancer la vision par ordinateur

---

### 3.2.1 LeNet-5 (1998)

- **Premier réseau convolutif** — Yann LeCun
- Utilisé pour reconnaissance de caractères **32×32 pixels**
- Activation : **tanh**

| Couche | Type | Cartes | Taille | Noyau | Pas |
|--------|------|--------|--------|-------|-----|
| In | Entrée | 1 | 32×32 | — | — |
| C1 | Convolution | 6 | 28×28 | 5×5 | 1 |
| S2 | Mean-pooling | 6 | 14×14 | 2×2 | 2 |
| C3 | Convolution | 16 | 10×10 | 5×5 | 1 |
| S4 | Mean-pooling | 16 | 5×5 | 2×2 | 2 |
| C5 | Convolution | 120 | 1×1 | 5×5 | 1 |
| F6 | FC | — | 84 | — | — |
| Out | FC | — | 10 | — | — |

---

### 3.2.2 AlexNet (2012)

- Ressemble à LeNet-5 mais **plus large et plus profonde**
- **Première** à empiler des couches de convolution **directement** sans pooling entre elles
- Activation : **ReLU** (plus rapide que tanh)
- Entrée : 3 (RGB) × 224×224

Points clés : 8 couches au total (5 conv + 3 FC), Dropout dans les couches FC, sortie 1000 classes Softmax.

---

### 3.2.3 ZF-Net (2013)

- Structure identique à AlexNet, mais **réglages ajustés** :
  - Filtre C1 : 11→**7**, stride : 4→**2**
  - Couches C3, C4, C5 : **384, 384, 256**
- Performances améliorées sans changer l'architecture

---

### 3.2.4 VGG-Net (2014)

- **Visual Geometry Group d'Oxford**
- Deux versions : **VGG-16** (13 conv + 3 FC) et **VGG-19** (16 conv + 3 FC)

**Points forts :**
- Concept de **blocs/modules**
- Utilise uniquement des **filtres 3×3** (remplace les grands filtres)
- Sous-échantillonnage uniquement par max pooling
- Taille de sortie après conv = taille entrée (padding same)
- Moins de paramètres grâce aux petits filtres

---

### 3.2.5 GoogLeNet / Inception (2014)

- **9 modules Inception** → utilisation des paramètres beaucoup plus efficace
- **22 couches**, mais **12× moins de paramètres** qu'AlexNet
- Toutes couches conv suivies de **ReLU**

**Module Inception :**
Le signal d'entrée est copié et envoyé en parallèle à 4 branches différentes :
1. Conv 1×1
2. Conv 1×1 → Conv 3×3
3. Conv 1×1 → Conv 5×5
4. Max-pooling 3×3 → Conv 1×1

→ Sorties **concaténées** en profondeur (depth concatenation)

**Avantages :**
- Détecte des motifs à **différentes échelles** (1×1, 3×3, 5×5)
- Les Conv 1×1 réduisent la dimensionnalité avant les grandes convolutions
- SAME padding → conserve hauteur et largeur

---

### 3.2.6 ResNet (2015)

**Idée de base :** Connexion de **raccourci (skip connection)** qui saute une ou plusieurs couches.

**Apprentissage résiduel :**
Au lieu d'apprendre H(x) directement, le réseau apprend le **résiduel** :
$$F(x) = H(x) - x \quad \Rightarrow \quad H(x) = F(x) + x$$

**Pourquoi c'est plus facile ?**
Il est plus simple d'apprendre la **différence entre entrée et sortie** que la transformation complète. La quantité H(x)−x est la variation, l'optimisation est plus simple.

**Structure d'une unité résiduelle :**
- 2 couches de convolution avec ReLU
- Noyaux 3×3, pas=1, remplissage same
- L'entrée x est ajoutée à la sortie → H(x) = F(x) + x

---

### 3.2.7 DenseNet (2016)

- **Extension de ResNet** (exploite les connexions raccourcies)
- **Toutes les couches** sont directement reliées entre elles (avant pooling)
- Réduction du nombre de filtres par couche
- Assure un **maximum de flux d'informations et de gradient**
- Architecture : Dense Blocks + Transition Layers (conv 1×1 + avg pooling 2×2)

---

## 4. Applications des CNN

- **Détection d'objets** : localisation + classification (ex. YOLO)
- **Détection de pose** : estimation de posture humaine
- **Segmentation sémantique** : classification pixel par pixel (ex. SegNet)
- **Voiture autonome** : traitement d'images infrarouge pour détection de piétons
