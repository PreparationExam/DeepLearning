# 🗒️ CHEAT SHEET — CNN

```
╔══════════════════════════════════════════════════════════════════════╗
║                  RÉSEAUX DE NEURONES CONVOLUTIFS                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ CONVOLUTION                                                          ║
║  Î(i,j) = ΣΣ K(n,m)·I(i+n,j+m)    [2D, 1 canal]                   ║
║  Î(i,j) = ΣΣΣ K(i,j,k)·I(i+n,j+m,k)  [multi-canal C]             ║
║  Filtre F×F, image N×M×C → volume F×F×C → sortie H×L×1             ║
╠══════════════════════════════════════════════════════════════════════╣
║ TAILLES DE SORTIE                                                    ║
║  Sans padding:  L = (M-F)/S + 1,   H = (N-F)/S + 1                 ║
║  Avec padding:  L = (M-F+2P)/S+1,  H = (N-F+2P)/S+1               ║
║  SAME (conserver):  P = (F-1)/2                                     ║
║  3×3→P=1  |  5×5→P=2  |  7×7→P=3                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ ARCHITECTURE TYPIQUE                                                 ║
║  Image → Conv+ReLU → Pool → Conv+ReLU → Pool → FC → Softmax        ║
║                                                                      ║
║  Conv    : N filtres → N feature maps (H×L×N)                       ║
║  ReLU    : max(0,z)  non-linéarité                                  ║
║  Pooling : max/avg pooling, stride=2 → taille/2                     ║
║  FC      : classifieur final (à la fin du CNN)                      ║
╠══════════════════════════════════════════════════════════════════════╣
║ FONCTIONS D'ACTIVATION                                               ║
║  ReLU       : max(0,z)           standard, rapide                   ║
║  Leaky ReLU : max(εz,z), ε≪1     dying ReLU fix                    ║
║  ELU        : max(α(eˣ-1),z)    dérivable partout                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ FILTRES CLASSIQUES (3×3)                                             ║
║  Identité    : [[0,0,0],[0,1,0],[0,0,0]]                            ║
║  Contours    : [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]                    ║
║  Netteté     : [[0,-1,0],[-1,5,-1],[0,-1,0]]                        ║
╠══════════════════════════════════════════════════════════════════════╣
║ ARCHITECTURES HISTORIQUES                                            ║
║  LeNet-5  (1998): 1er CNN, tanh, images 32×32                       ║
║  AlexNet  (2012): ReLU+Dropout, conv empilées, 1000 classes         ║
║  ZF-Net   (2013): AlexNet filtre 11→7, stride 4→2                   ║
║  VGG-16   (2014): blocs, filtres 3×3 uniquement, SAME padding       ║
║  GoogLeNet(2014): 9 modules Inception, 22 couches, 12× < AlexNet   ║
║  ResNet   (2015): skip connections, H(x)=F(x)+x résiduel           ║
║  DenseNet (2016): toutes couches reliées, flux gradient maximal     ║
╠══════════════════════════════════════════════════════════════════════╣
║ RESNET — APPRENTISSAGE RÉSIDUEL                                      ║
║  Standard : apprendre H(x)                                          ║
║  ResNet   : apprendre F(x) = H(x)-x  → H(x) = F(x)+x              ║
║  Skip connection : ajouter x directement à la sortie du bloc        ║
║  Unité résiduelle : 2 Conv 3×3 + ReLU + skip                       ║
╠══════════════════════════════════════════════════════════════════════╣
║ MODULE INCEPTION (GoogLeNet)                                         ║
║  4 branches parallèles sur même entrée :                            ║
║  [Conv 1×1] [Conv 1×1 → Conv 3×3] [Conv 1×1 → Conv 5×5]           ║
║  [MaxPool 3×3 → Conv 1×1]                                           ║
║  → Concaténation en profondeur (depth concat)                       ║
║  → Détecte patterns à échelles 1×1, 3×3, 5×5 simultanément         ║
╠══════════════════════════════════════════════════════════════════════╣
║ PARAMÈTRES D'UN FILTRE                                               ║
║  1 filtre F×F×C : F×F×C poids                                       ║
║  N filtres       : N×F×F×C poids + N biais                          ║
║  Ex: 32 filtres 3×3 sur RGB: 32×3×3×3 + 32 = 896 params            ║
╠══════════════════════════════════════════════════════════════════════╣
║ POOLING                                                              ║
║  Max pooling   : max de chaque zone → saillance                     ║
║  Avg pooling   : moyenne de chaque zone → lissage                   ║
║  Filtre 2×2, S=2 → divise hauteur et largeur par 2                  ║
║  Augmente invariance : rotation, translation, symétrie, échelle     ║
╠══════════════════════════════════════════════════════════════════════╣
║ IMAGENET / ILSVRC                                                    ║
║  15M images, 22K catégories (dataset complet)                       ║
║  ILSVRC: 1000 catégories, 1000 img/cat, depuis 2010                 ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Exemple rapide de calcul de taille

```
Entrée: 32×32×3
Conv: 8 filtres 5×5, S=1, P=2
  → H = (32-5+2×2)/1 + 1 = 32  ← même taille (SAME)
  → sortie: 32×32×8

MaxPool: 2×2, S=2
  → 16×16×8

Conv: 16 filtres 3×3, S=1, P=1
  → 16×16×16

MaxPool: 2×2, S=2
  → 8×8×16

FC: 8×8×16 = 1024 → aplatir → couche FC → Softmax
```
