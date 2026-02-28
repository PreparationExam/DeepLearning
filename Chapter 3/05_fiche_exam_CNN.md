# 🎯 FICHE EXAM — CNN (Ce qui va tomber)

---

## 🔴 ULTRA-PRIORITAIRE

### 1. Calcul de taille de sortie — à faire les yeux fermés

**Sans padding :** L = (M-F)/S + 1
**Avec padding :** L = (M-F+2P)/S + 1

**Exemples types :**
- Entrée 7×7, filtre 3×3, S=1, P=0 → (7-3)/1 + 1 = **5×5**
- Entrée 7×7, filtre 3×3, S=2, P=0 → (7-3)/2 + 1 = **3×3**
- Entrée 7×7, filtre 3×3, S=1, P=1 → (7-3+2)/1 + 1 = **7×7** (SAME)
- Entrée 32×32, filtre 5×5, S=1, P=2 → (32-5+4)/1+1 = **32×32** (SAME)

---

### 2. Padding SAME : formule et valeurs

P = (F-1)/2
- 3×3 → P=**1**, 5×5 → P=**2**, 7×7 → P=**3**

---

### 3. Architecture CNN standard (ordre !)

```
Image → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Softmax → Classes
```

---

### 4. Nombre de paramètres d'une couche conv

N filtres F×F sur C canaux = **N × F × F × C + N** (biais)

Ex : 64 filtres 3×3 sur image RGB : 64×3×3×3 + 64 = **1 792 paramètres**

---

## 🟠 TRÈS PROBABLE

### 5. Résumé des architectures (tableau à connaître)

| Réseau | Clé à retenir |
|--------|--------------|
| LeNet-5 | 1er CNN, tanh, 32×32 |
| AlexNet | ReLU+Dropout, conv empilées |
| ZF-Net | AlexNet réglé (filtre 11→7) |
| VGG | Filtres 3×3 uniquement, blocs |
| GoogLeNet | Modules Inception, 22 couches, 12× < AlexNet |
| ResNet | H(x)=F(x)+x, skip connections |
| DenseNet | Toutes couches connectées |

---

### 6. ResNet — formule et justification

H(x) = F(x) + x

Pourquoi c'est mieux : apprendre F(x)=H(x)-x (le **résiduel**) est plus simple que H(x) directement. Résout le vanishing gradient pour les réseaux profonds.

---

### 7. Module Inception — 4 branches

1. Conv 1×1
2. Conv 1×1 → Conv 3×3
3. Conv 1×1 → Conv 5×5
4. MaxPool 3×3 → Conv 1×1

→ Concaténation en profondeur → détecte motifs à plusieurs échelles

---

### 8. Pooling — types et effets

| Type | Opération | Utilité |
|------|-----------|---------|
| Max | max(zone) | Saillance, le + utilisé |
| Average | mean(zone) | Lissage |
| Les deux | — | Invariance rotation/translation |

---

## 🟡 PROBABLE

### 9. Définitions à rédiger

- **Carte de caractéristiques :** sortie d'une convolution, représente les réponses du filtre à chaque position
- **Stride :** pas de déplacement du filtre, contrôle la taille de sortie
- **Padding :** ajout de pixels (zéros) en bordure pour permettre la convolution sur les bords
- **ReLU :** max(0,z), introduction de non-linéarité, évite saturation vs sigmoïde
- **Max Pooling :** sous-échantillonnage par valeur max → invariance aux transformations
- **Skip connection :** connexion directe d'entrée → sortie d'un bloc, permet apprentissage résiduel

---

### 10. Filtres classiques

- **Identité :** [[0,0,0],[0,1,0],[0,0,0]] → image inchangée
- **Contours :** [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] → détecte les bords
- **Netteté :** [[0,-1,0],[-1,5,-1],[0,-1,0]] → renforce les détails

---

## ⚠️ Erreurs classiques à éviter

- Oublier le **+1** dans la formule de taille de sortie
- Confondre **SAME** (sortie = entrée) et **VALID** (sortie < entrée)
- Croire que S=3 sur une entrée 7×7 avec filtre 3×3 est possible [(7-3)/3 = 1.33 → impossible !]
- Confondre **profondeur du filtre** (= C canaux de l'entrée) et **nb de filtres** (= profondeur sortie)
- Oublier que la **couche FC est à la fin** (pas au début ni au milieu)
- Penser que ResNet "saute" des couches — non, il **ajoute** l'entrée à la sortie
