# 💪 50 Exercices & Problèmes avec Solutions — CNN

---

## PARTIE 1 — QCM & Concepts (1-15)

**Ex 1.** Une image 12×12×3 est convoluée avec un filtre 3×3, stride=1, P=0. Taille de sortie ?
**✅** L = (12-3)/1 + 1 = **10×10×1** (par filtre)

---

**Ex 2.** Même image, on applique 16 filtres. Taille de sortie totale ?
**✅** **10×10×16** — chaque filtre produit une carte 10×10, et 16 filtres → profondeur 16.

---

**Ex 3.** Image 28×28×1, filtre 5×5, S=1, P=2. Taille de sortie ?
**✅** L = (28-5+2×2)/1 + 1 = (28-5+4)/1+1 = 27+1 = **28×28** (SAME)

---

**Ex 4.** Image 64×64×3, filtre 3×3, S=2, P=1. Taille de sortie ?
**✅** L = (64-3+2×1)/2 + 1 = 63/2 + 1 = 31.5 + 1 → arrondi bas = **32×32**
Note : L = floor((64-3+2)/2) + 1 = floor(63/2)+1 = 31+1 = **32×32**

---

**Ex 5.** Image 7×7×C, filtre 3×3, S=3. Est-ce possible ?
**✅ Non.** (7-3)/3 = 4/3 ≈ 1.33 — pas un entier. Convolution **impossible** avec S=3 ici.

---

**Ex 6.** Quel padding ajouter à un filtre 7×7 pour obtenir SAME ?
**✅** P = (7-1)/2 = **P = 3**

---

**Ex 7.** Max pooling 2×2, stride=2 sur une carte 14×14. Taille après pooling ?
**✅** L = (14-2)/2 + 1 = 6+1 = **7×7** — taille divisée par 2.

---

**Ex 8.** Une couche conv a 32 filtres 3×3 sur une entrée à 64 canaux. Combien de paramètres (avec biais) ?
**✅** 32 × 3 × 3 × 64 + 32 = 18 432 + 32 = **18 464 paramètres**

---

**Ex 9.** VGG-16 utilise des filtres de quelle taille exclusivement ?
**✅ 3×3** uniquement. C'est sa caractéristique principale.

---

**Ex 10.** Vrai ou Faux : Dans un CNN, le Max Pooling contient des paramètres apprenables.
**✅ Faux.** Le pooling n'a **aucun paramètre** entraînable — c'est une opération fixe (max ou moyenne).

---

**Ex 11.** Quelle activation est utilisée après chaque convolution dans GoogLeNet ?
**✅ ReLU** — "Toutes les couches de convolution sont suivies de ReLU."

---

**Ex 12.** Qu'est-ce que le "dying ReLU" ?
**✅** Neurones dont la sortie est **toujours 0** (entrées toutes négatives) → gradient nul → plus d'apprentissage. Solution : Leaky ReLU.

---

**Ex 13.** AlexNet a quelle taille d'entrée ?
**✅** 3 (RGB) × **224×224** pixels.

---

**Ex 14.** Quelle est la formule du résiduel dans ResNet ?
**✅** H(x) = F(x) + x, où F(x) = H(x) - x est le **résiduel**. La skip connection ajoute x directement à la sortie F(x).

---

**Ex 15.** Vrai ou Faux : Dans DenseNet, chaque couche reçoit uniquement les sorties de la couche précédente.
**✅ Faux.** Dans DenseNet, chaque couche reçoit les feature maps de **toutes les couches précédentes** (avant pooling).

---

## PARTIE 2 — Calculs Numériques (16-30)

**Ex 16.** Calculez manuellement la convolution 3×3 suivante.

Image 3×3 :
```
1 2 3
4 5 6
7 8 9
```
Filtre identité [[0,0,0],[0,1,0],[0,0,0]], P=1, S=1. Sortie au pixel central (1,1) ?

**✅** Le pixel central (1,1) de l'image est 5. Le filtre identité extrait le pixel central directement : **sortie = 5**.

---

**Ex 17.** Calculez la convolution du pixel central (1,1) avec le filtre de détection de contours.

Image 3×3 :
```
0 0 0
0 5 0
0 0 0
```
Filtre : [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

**✅**
```
(-1×0)+(-1×0)+(-1×0)
+(-1×0)+(8×5)+(-1×0)
+(-1×0)+(-1×0)+(-1×0)
= 0 + 40 + 0 = 40
```

---

**Ex 18.** Appliquez le filtre de netteté [[0,-1,0],[-1,5,-1],[0,-1,0]] au pixel central d'une image 3×3 :
```
2 3 1
1 4 2
3 2 1
```

**✅**
```
(0×2)+(-1×3)+(0×1)
+(-1×1)+(5×4)+(-1×2)
+(0×3)+(-1×2)+(0×1)
= 0-3+0-1+20-2+0-2+0 = 12
```

---

**Ex 19.** Max pooling 2×2, S=2 sur la matrice suivante :
```
1  3  2  4
5  7  8  6
3  9  1  2
4  6  5  8
```

**✅** On divise en 4 zones 2×2 :
- Zone haut-gauche [1,3,5,7] → max = **7**
- Zone haut-droite [2,4,8,6] → max = **8**
- Zone bas-gauche [3,9,4,6] → max = **9**
- Zone bas-droite [1,2,5,8] → max = **8**

Sortie 2×2 :
```
7 8
9 8
```

---

**Ex 20.** Average pooling 2×2, S=2 sur la même matrice.

**✅**
- Zone haut-gauche : (1+3+5+7)/4 = **4**
- Zone haut-droite : (2+4+8+6)/4 = **5**
- Zone bas-gauche : (3+9+4+6)/4 = **5.5**
- Zone bas-droite : (1+2+5+8)/4 = **4**

Sortie :
```
4    5
5.5  4
```

---

**Ex 21.** LeNet-5 : après C1 (conv 5×5, S=1, P=0), quelle est la taille de sortie si l'entrée est 32×32 ?

**✅** L = (32-5)/1 + 1 = **28×28** avec 6 cartes → **28×28×6** ✓

---

**Ex 22.** Après S2 (mean-pool 2×2, S=2) dans LeNet-5. Taille ?

**✅** L = (28-2)/2 + 1 = **14×14×6** ✓

---

**Ex 23.** Combien de paramètres pour une couche conv avec 8 filtres 5×5 sur une entrée RGB ?

**✅** 8 × 5 × 5 × 3 + 8 = 600 + 8 = **608 paramètres**

---

**Ex 24.** Vous avez une entrée 224×224×3. Après conv 11×11, S=4, P=0 (comme AlexNet C1) combien de filtres pour obtenir 55×55×96 ?

**✅** Taille : (224-11)/4 + 1 = 213/4 + 1 ≈ 53.25 +1 → avec SAME padding : **55×55**. Nombre de filtres = **96** ✓

---

**Ex 25.** Après le MaxPool S2 d'AlexNet (96×55×55, pool 3×3, S=2) : taille de sortie ?

**✅** L = (55-3)/2 + 1 = 52/2 + 1 = 26 + 1 = **27×27×96** ✓

---

**Ex 26.** Une couche FC reçoit une carte aplatie de 7×7×512 (VGG). Combien d'entrées ?

**✅** 7 × 7 × 512 = **25 088 entrées** → connectées à 4096 neurones = 25088×4096 + 4096 ≈ **102M paramètres** (couche énorme !)

---

**Ex 27.** Calculez la sortie ReLU sur le vecteur [-3, 0, 2, -1, 5].

**✅** ReLU: max(0,x) pour chaque élément → [**0, 0, 2, 0, 5**]

---

**Ex 28.** Leaky ReLU (ε=0.1) sur le même vecteur [-3, 0, 2, -1, 5].

**✅** Pour x<0 : 0.1×x, sinon x → [**-0.3, 0, 2, -0.1, 5**]

---

**Ex 29.** Image 32×32, 5 opérations successives : Conv 3×3 SAME, MaxPool 2×2 S=2, Conv 3×3 SAME, MaxPool 2×2 S=2, Conv 3×3 SAME. Taille spatiale finale ?

**✅**
- Conv SAME : 32×32 (inchangé)
- MaxPool : 32/2 = 16×16
- Conv SAME : 16×16
- MaxPool : 16/2 = 8×8
- Conv SAME : **8×8**

---

**Ex 30.** Un module Inception reçoit une entrée 28×28×256. Les 4 branches (toutes SAME) produisent respectivement 64, 128, 32, 32 cartes. Taille de sortie après concaténation ?

**✅** Toutes les branches conservent 28×28 (SAME), profondeur totale = 64+128+32+32 = **256**. Sortie : **28×28×256**

---

## PARTIE 3 — Problèmes Approfondis (31-45)

**Ex 31.** Décrivez couche par couche l'architecture d'un CNN simple pour classer des images CIFAR-10 (32×32×3, 10 classes).

**✅** Architecture possible :
```
Entrée: 32×32×3
Conv 3×3, 32 filtres, S=1, P=1 → 32×32×32 + ReLU
MaxPool 2×2, S=2 → 16×16×32
Conv 3×3, 64 filtres, S=1, P=1 → 16×16×64 + ReLU
MaxPool 2×2, S=2 → 8×8×64
Conv 3×3, 128 filtres, S=1, P=1 → 8×8×128 + ReLU
MaxPool 2×2, S=2 → 4×4×128
Flatten: 4×4×128 = 2048
FC: 2048 → 256 + ReLU
FC: 256 → 10 + Softmax
```

---

**Ex 32.** Pourquoi VGG utilise-t-il plusieurs couches 3×3 au lieu d'une couche de grande taille ?

**✅** Deux convolutions 3×3 successives ont le même **champ réceptif** qu'une convolution 5×5, mais avec **moins de paramètres** et **plus de non-linéarité** :
- 2 × 3×3 = 18 poids vs 1 × 5×5 = 25 poids (par canal)
- En plus, 2 activations ReLU vs 1 → plus de complexité non-linéaire

---

**Ex 33.** Expliquez pourquoi les CNN sont dits "invariants aux translations".

**✅** Grâce au **partage de poids** (même filtre appliqué partout) et au **pooling** : si un objet se décale légèrement dans l'image, le filtre le détectera à la nouvelle position → même réponse. Le pooling amplifie cette invariance en réduisant la résolution spatiale.

---

**Ex 34.** Calculez le nombre total de paramètres de LeNet-5.

**✅**
- C1 : 6 filtres 5×5×1 + biais = 6×25+6 = **156**
- S2 : 6 coefficients de pooling + 6 biais = **12**
- C3 : 16 filtres 5×5×6 + biais = 16×150+16 = **2416**
- S4 : 16×2 = **32**
- C5 : 120 filtres 5×5×16 = 120×400+120 = **48 120**
- F6 : 84×120+84 = **10 164**
- Out : 10×84+10 = **850**

Total ≈ **61 750 paramètres**

---

**Ex 35.** Comparez ResNet et DenseNet.

**✅**

| Critère | ResNet | DenseNet |
|---------|--------|---------|
| Connexions | x + F(x) (couche précédente) | Toutes les couches précédentes |
| Opération | Addition | Concaténation |
| Paramètres | Plus | Moins (partage de features) |
| Flux gradient | Amélioré | Maximal |

---

**Ex 36.** Une couche de pooling 2×2, S=2 réduit la taille de moitié. Que se passe-t-il si on applique 3 pooling successifs sur une image 64×64 ?

**✅**
- Après pool 1 : 32×32
- Après pool 2 : 16×16
- Après pool 3 : **8×8**

Réduction de 64 à 8 = facteur 8 → 64 fois moins de pixels.

---

**Ex 37.** Expliquez le rôle des convolutions 1×1 dans GoogLeNet.

**✅** Les convolutions 1×1 (aussi appelées **bottleneck**) réduisent la **profondeur (nb de canaux)** sans changer les dimensions spatiales. Exemple : 256 canaux → 32 canaux via 1×1 conv → puis 3×3 conv sur 32 canaux au lieu de 256 → drastiquement moins de paramètres.

---

**Ex 38.** Pourquoi applique-t-on la couche FC après avoir "aplati" (flatten) les feature maps ?

**✅** La couche FC attend un **vecteur 1D**. Les feature maps sont des tenseurs 3D (H×L×C). Le flatten transforme H×L×C → vecteur de taille H×L×C, que la FC peut traiter normalement.

---

**Ex 39.** Décrivez l'impact du stride sur la taille de sortie et sur l'apprentissage.

**✅**
- S grand → sortie plus petite → perte d'information spatiale mais vision plus globale
- S=1 → sortie plus grande → plus de détails préservés
- Dans les réseaux modernes, on préfère parfois une conv avec S=2 à un pooling 2×2 car la conv apprend à **sous-échantillonner** de manière optimale (poids apprenables)

---

**Ex 40.** Quelle est la différence entre "caractéristiques bas niveau" et "haut niveau" dans un CNN ?

**✅**
- **Bas niveau :** premières couches → bords, coins, textures, gradients locaux. Ces caractéristiques sont génériques et se retrouvent dans toutes les images
- **Haut niveau :** dernières couches → parties d'objets (roues, yeux), puis objets entiers, scènes. Très spécifiques à la tâche

---

**Ex 41.** Pourquoi le zero-padding ne distord-il pas trop l'image ?

**✅** Pour un filtre de taille F, on ajoute seulement P = (F-1)/2 zéros. Les pixels du bord de l'image réelle sont quand même traités correctement — les zéros n'influencent que les calculs aux extrémités. L'effet est minimal sur le contenu central.

---

**Ex 42.** Architecture VGG-16 : elle a 13 couches conv et 3 FC. Combien de couches au total ?

**✅** 13 conv + 3 FC = **16 couches** avec paramètres → d'où le nom VGG-**16** ✓ (+ les pooling qui n'ont pas de paramètres)

---

**Ex 43.** Quelle est la taille du champ réceptif de 3 convolutions 3×3 empilées ?

**✅** Chaque conv 3×3 ajoute 2 pixels de contexte de chaque côté.
- 1 conv 3×3 → champ réceptif 3×3
- 2 conv 3×3 → champ réceptif 5×5
- 3 conv 3×3 → champ réceptif **7×7**

C'est pour ça que VGG remplace un seul filtre 7×7 par 3 filtres 3×3 : même vision, moins de paramètres.

---

**Ex 44.** Expliquez pourquoi l'invariance aux transformations du pooling est importante pour la reconnaissance d'images.

**✅** Dans la réalité, le même objet peut apparaître à différentes positions, orientations ou tailles. Sans invariance, le réseau devrait mémoriser chaque variante. Avec le pooling, des petites variations de position ne changent pas la réponse → le réseau **généralise** mieux. C'est essentiel pour la robustesse en conditions réelles.

---

**Ex 45.** Comparez GoogLeNet et AlexNet en termes de paramètres et de performance.

**✅**

| Critère | AlexNet | GoogLeNet |
|---------|---------|-----------|
| Couches | 8 | 22 |
| Paramètres | ~60M | ~5M (12× moins) |
| Activation | ReLU | ReLU |
| Innovation | 1ère à empiler conv | Modules Inception |
| Performance | ILSVRC 2012 | ILSVRC 2014 |

GoogLeNet utilise les paramètres **beaucoup plus efficacement** grâce aux modules Inception et aux convolutions 1×1.

---

## PARTIE 4 — Synthèse (46-50)

**Ex 46.** Complétez le tableau de tailles pour un CNN :

| Couche | Entrée | Opération | Sortie |
|--------|--------|-----------|--------|
| 1 | 224×224×3 | Conv 11×11, S=4, P=0, 96 filtres | ? |
| 2 | ? | MaxPool 3×3, S=2 | ? |
| 3 | ? | Conv 5×5, P=2, 256 filtres | ? |

**✅**
- Couche 1 : (224-11)/4 +1 = **55×55×96**
- Couche 2 : (55-3)/2 +1 = **27×27×96**
- Couche 3 : (27-5+4)/1+1 = **27×27×256** (SAME)

---

**Ex 47.** Résumez les avantages des CNN par rapport aux PMC pour le traitement d'images en 5 points.

**✅**
1. **Partage de poids** → vastement moins de paramètres
2. **Localité spatiale** → les features locales sont capturées par petits filtres
3. **Invariance** aux translations grâce au pooling
4. **Hiérarchie** automatique de features (bas → haut niveau)
5. **Efficacité** computationnelle : une conv 3×3 sur 224×224 vs couche FC complète

---

**Ex 48.** Décrivez le principe de la segmentation sémantique avec CNN (ex. SegNet).

**✅** La segmentation sémantique classe **chaque pixel** de l'image. Architecture encodeur-décodeur :
- **Encodeur** : CNN classique (Conv+Pool) → compresse l'image en représentation compacte
- **Décodeur** : upsampling progressif → reconstruit la carte de segmentation pixel par pixel
- **Softmax** par pixel → probabilité d'appartenir à chaque classe
- Sortie : image de même taille que l'entrée, chaque pixel coloré selon sa classe

---

**Ex 49.** Pourquoi dit-on que ResNet "facilite l'apprentissage" ?

**✅**
1. **Problème identité :** si F(x)=0, H(x)=x → réseau apprend facilement à ne rien faire
2. **Gradient direct :** la skip connection crée un chemin direct pour le gradient → pas de vanishing
3. **Résiduel faible :** H(x)−x est souvent proche de 0 → optimisation plus stable
4. Permet d'entraîner des réseaux de **100, 152, voire 1000 couches**

---

**Ex 50.** Cas pratique : vous voulez construire un CNN pour détecter des tumeurs dans des IRM (images 256×256 en niveaux de gris, 2 classes). Proposez une architecture et justifiez chaque choix.

**✅** Architecture proposée :
```
Entrée: 256×256×1 (1 canal = niveaux de gris)
Conv 3×3, 32 filtres, P=1, ReLU → 256×256×32
MaxPool 2×2, S=2 → 128×128×32
Conv 3×3, 64 filtres, P=1, ReLU → 128×128×64
MaxPool 2×2, S=2 → 64×64×64
Conv 3×3, 128 filtres, P=1, ReLU → 64×64×128
MaxPool 2×2, S=2 → 32×32×128
Conv 3×3, 256 filtres, P=1, ReLU → 32×32×256
GlobalAvgPool → 256
FC: 256 → 128, ReLU
Dropout 0.5
FC: 128 → 2, Softmax
```

**Justifications :**
- Entrée 1 canal (IRM en N&G)
- Petits filtres 3×3 → efficace, moins de paramètres
- Profondeur croissante (32→256) → hiérarchie de features
- Global Average Pooling → réduit overfitting vs Flatten + FC
- Dropout avant dernière FC → régularisation (données médicales limitées)
- 2 neurones Softmax → classification binaire (tumeur/sain)
- Data augmentation recommandée : rotation, flip, zoom (données médicales rares)
