# ⚡ Résumé Court — CNN (L'essentiel)

---

## 1. Convolution en 3 lignes
- Filtre K (F×F) glisse sur image → produit scalaire à chaque position → **feature map**
- Formule : Î(i,j) = ΣΣ K(n,m)·I(i+n, j+m)
- Image multi-canaux C → filtre F×F×C → 1 feature map

---

## 2. Paramètres clés

| Paramètre | Formule sortie | Notes |
|-----------|---------------|-------|
| **Stride S** | L = (M-F)/S + 1 | S=1 ou 2, S↑ → taille↓ |
| **Padding P** | L = (M-F+2P)/S + 1 | P=(F-1)/2 pour SAME |
| **SAME** | sortie = entrée | auto-padding |
| **VALID** | sortie < entrée | pas de padding |

**Padding par filtre :** 3×3→P=1, 5×5→P=2, 7×7→P=3

---

## 3. Architecture CNN
```
Image → [Conv → ReLU] → [Pooling] → [Conv → ReLU] → [Pooling] → [FC] → Softmax
```

| Couche | Rôle |
|--------|------|
| **Conv** | Extraire features, N filtres → N cartes |
| **ReLU** | Non-linéarité (g=max(0,z)) |
| **Pooling** | Réduire taille, invariance aux transfo |
| **FC** | Classification (à la fin) |

---

## 4. Pourquoi CNN et pas PMC ?
- Image 224×224×3 = 150K pixels → trop de poids en FC
- CNN : **partage de poids** (même filtre sur toute l'image)
- Premières couches → **bas niveau** (bords, textures)
- Dernières couches → **haut niveau** (objets)

---

## 5. ReLU et variantes
- **ReLU** : max(0,z) — standard, rapide
- **Leaky ReLU** : max(εz,z) — résout dying ReLU
- **ELU** : max(α(eˣ−1),z) — dérivable partout

---

## 6. Pooling
- **Max pooling** : max de la zone → garde le plus saillant ✓ (le plus utilisé)
- **Avg pooling** : moyenne de la zone → plus lisse
- Filtre 2×2, stride=2 → divise la taille par 2

---

## 7. Architectures historiques (à retenir)

| Réseau | Année | Innovation clé |
|--------|-------|---------------|
| **LeNet-5** | 1998 | Premier CNN, tanh |
| **AlexNet** | 2012 | ReLU, Dropout, conv empilées |
| **ZF-Net** | 2013 | Réglage AlexNet (filtre 11→7) |
| **VGG** | 2014 | Blocs, filtres 3×3 uniquement |
| **GoogLeNet** | 2014 | Modules Inception, 12× moins params |
| **ResNet** | 2015 | Skip connections, apprentissage résiduel |
| **DenseNet** | 2016 | Toutes couches connectées entre elles |

---

## 8. ResNet en 3 lignes
- Skip connection : H(x) = F(x) + x
- Apprendre F(x) = résiduel plus facile que H(x) direct
- Résout le vanishing gradient pour les réseaux très profonds

## 9. Inception en 3 lignes
- 4 branches en parallèle : 1×1, 3×3, 5×5, max-pool
- Sorties concaténées en profondeur
- Détecte patterns à plusieurs échelles simultanément
