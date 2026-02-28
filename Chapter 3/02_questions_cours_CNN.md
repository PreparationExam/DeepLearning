# ❓ Questions de Cours — CNN (Toutes les définitions)

---

**Q1. Qu'est-ce qu'une convolution d'image ?**
Opération **linéaire** entre une image I et un filtre (kernel) K de taille F×F, qui produit une nouvelle image appelée **carte de caractéristiques (feature map)**. Chaque pixel de la carte est la somme pondérée des poids du filtre × pixels voisins correspondants.

$$\hat{I}(i,j) = \sum_{n=0}^{F-1}\sum_{m=0}^{F-1} K(n,m) \cdot I(i+n, j+m)$$

---

**Q2. Qu'est-ce qu'une carte de caractéristiques (feature map) ?**
La sortie Î d'une opération de convolution. C'est une image transformée qui représente les réponses du filtre à chaque position de l'image d'entrée.

---

**Q3. Comment un filtre F×F est-il appliqué à une image multi-canaux (C canaux) ?**
Le filtre devient un volume **F×F×C**. La convolution est étendue sur les C canaux :
$$\hat{I}(i,j) = \sum_{k=1}^{C}\sum_{n=0}^{F-1}\sum_{m=0}^{F-1} K(i,j,k) \cdot I(i+n, j+m, k)$$

---

**Q4. Qu'est-ce que la stride (pas de convolution) ? Quel est son effet sur la taille de sortie ?**
La **stride S** est le nombre de pixels dont la fenêtre se déplace après chaque convolution. La taille de sortie est :
$$L = \frac{M-F}{S}+1 \quad \text{et} \quad H = \frac{N-F}{S}+1$$
Plus S est grand, plus la sortie est **petite**. S généralement = 1 ou 2.

---

**Q5. Quel est le problème des pixels en bordure et comment le résout-on ?**
La convolution est **impossible** pour les pixels en bordure (pas assez de voisins). Solution : le **padding** (remplissage) qui agrandit l'image avant convolution.
- **Zero-padding :** ajouter P zéros de chaque côté
- **Duplication :** dupliquer les bords

---

**Q6. Quelle est la formule de la taille de sortie avec padding P et stride S ?**
$$L = \frac{M-F+2P}{S}+1 \quad \text{et} \quad H = \frac{N-F+2P}{S}+1$$
Pour conserver la même taille (SAME), on utilise P = (F-1)/2.

---

**Q7. Quel padding utiliser pour un filtre 3×3 ? 5×5 ? 7×7 ?**
Formule : P = (F-1)/2
- Filtre 3×3 → P = **1**
- Filtre 5×5 → P = **2**
- Filtre 7×7 → P = **3**

---

**Q8. Qu'est-ce qu'un CNN ? Pourquoi l'utiliser pour les images plutôt qu'un PMC ?**
Un **réseau de neurones convolutif** est un réseau utilisant des opérations de convolution au lieu de connexions full. Avantages vs PMC :
- Pas besoin de traiter l'image entière en une fois (trop de poids)
- **Partage de poids** : même filtre sur toute l'image → moins de paramètres
- Détecte des motifs locaux similaires partout dans l'image

---

**Q9. Qu'est-ce que le partage de poids dans un CNN ?**
Les neurones d'une **même couche convolutive** partagent les mêmes poids (le même filtre). Ce filtre est appliqué sur toute l'image → réduit massivement le nombre de paramètres.

---

**Q10. Quelles caractéristiques extraient les premières vs dernières couches d'un CNN ?**
- **Premières couches :** caractéristiques **bas niveau et locales** (bords, textures, coins) — similaires partout dans l'image
- **Dernières couches :** formes **haut niveau et globales** (objets, visages, scènes)

---

**Q11. Quels sont les 4 types de couches d'un CNN classique ?**
1. **Couche de convolution** : extraction de features par filtres
2. **Fonction d'activation** (ReLU) : introduction de non-linéarité
3. **Couche de pooling** : réduction de dimension, invariance
4. **Couche totalement connectée (FC)** : classification finale

---

**Q12. Qu'est-ce que la fonction ReLU ? Pourquoi l'utilise-t-on dans les CNN ?**
ReLU (Rectified Linear Unit) : g(z) = max(0, z)
Elle introduit de la **non-linéarité** dans le réseau, permettant d'apprendre des représentations complexes. Elle est rapide, simple et évite le problème de saturation des gradients comparé à sigmoïde/tanh.

---

**Q13. Quelles sont les variantes de ReLU et leurs avantages respectifs ?**
| Variante | Formule | Avantage |
|----------|---------|---------|
| ReLU | max(0, z) | Simple, biologique |
| Leaky ReLU | max(εz, z), ε≪1 | Résout dying ReLU |
| ELU | max(α(eˣ−1), z), α≪1 | Dérivable partout |

---

**Q14. Qu'est-ce que le "dying ReLU" ?**
Problème où des neurones ReLU deviennent **définitivement inactifs** (sortie toujours 0) si leurs entrées sont systématiquement négatives → le gradient est 0 → ils n'apprennent plus. Solution : Leaky ReLU ou ELU.

---

**Q15. Qu'est-ce que le pooling ? Quel est son rôle ?**
Opération de **sous-échantillonnage** qui découpe la feature map en zones sans chevauchement et les réduit :
- **Max pooling :** valeur maximale de chaque zone
- **Average/Mean pooling :** valeur moyenne
Rôles : réduire la taille des cartes + augmenter l'**invariance** aux transformations.

---

**Q16. Quelle est la différence entre max pooling et average pooling ?**
- **Max pooling** : prend la valeur max → conserve les features les plus saillantes (plus courant)
- **Average pooling** : prend la moyenne → lissage, moins agressif, perd moins d'information

---

**Q17. Qu'est-ce qu'une couche FC (Fully Connected) dans un CNN ? Où se trouve-t-elle ?**
Couche où **chaque entrée est connectée à tous les neurones** (identique au PMC). Présente à la **fin** des CNN pour la classification (calcul des scores de classe). Cas particulier : taille filtre = taille entrée → équivalent FC.

---

**Q18. Qu'est-ce qu'ImageNet ? Qu'est-ce que l'ILSVRC ?**
- **ImageNet :** dataset de 15 millions d'images et 22 000 catégories
- **ILSVRC** (ImageNet Large Scale Visual Recognition Challenge) : compétition annuelle depuis 2010, 1 000 catégories, 1 000 images/catégorie, 50 000 validation, 150 000 test

---

**Q19. Qu'est-ce que LeNet-5 ? Quelles sont ses caractéristiques ?**
Premier réseau convolutif (Yann LeCun, 1998). Reconnaît des caractères 32×32. Structure : Entrée → Conv(tanh) → Mean-pool → Conv(tanh) → Mean-pool → Conv → FC(tanh) → FC(RBF). Activation : tanh.

---

**Q20. Quelle innovation majeure AlexNet a-t-elle introduite ?**
- Première à **empiler des couches conv directement** sans pooling intermédiaire
- Utilise **ReLU** au lieu de tanh
- Utilise le **Dropout** dans les FC
- Plus large et profonde que LeNet

---

**Q21. Quelle est la différence entre ZF-Net et AlexNet ?**
ZF-Net garde la même structure mais change : filtre C1 11→**7**, stride C1 4→**2**, couches C3/C4/C5 ajustées à 384/384/256. Les performances s'améliorent sans changer l'architecture de base.

---

**Q22. Quels sont les points forts de VGG-Net ?**
- Concept de **blocs** répétables
- Utilise **uniquement des filtres 3×3** (remplace les grands filtres par plusieurs petits)
- Sous-échantillonnage par max pooling uniquement
- Taille sortie = taille entrée après conv (SAME padding)
- Moins de paramètres, plus profond

---

**Q23. Qu'est-ce qu'un module Inception (GoogLeNet) ?**
Sous-réseau qui applique **en parallèle** différentes tailles de convolution (1×1, 3×3, 5×5) et un max pooling, puis **concatène** les sorties en profondeur. Permet de détecter des motifs à différentes échelles simultanément. GoogLeNet a 9 de ces modules.

---

**Q24. Pourquoi GoogLeNet a-t-il 12× moins de paramètres qu'AlexNet malgré ses 22 couches ?**
Grâce aux **convolutions 1×1** (bottleneck) qui réduisent la dimensionnalité avant les convolutions 3×3 et 5×5, et au partage plus efficace via les modules Inception.

---

**Q25. Qu'est-ce que l'apprentissage résiduel (ResNet) ?**
Au lieu d'apprendre H(x) directement, le réseau apprend le **résiduel** F(x) = H(x) - x. Une connexion de **raccourci (skip connection)** ajoute l'entrée x à la sortie :
$$H(x) = F(x) + x$$
Plus simple à optimiser car on apprend juste la **différence** entre entrée et sortie.

---

**Q26. Pourquoi les skip connections de ResNet résolvent-elles le problème du gradient vanishing ?**
Les skip connections créent un **chemin direct** pour le gradient de la sortie vers les couches profondes, sans passer par toutes les transformations. Le gradient peut circuler directement → apprentissage efficace même avec des réseaux très profonds (100+ couches).

---

**Q27. Qu'est-ce que DenseNet et en quoi diffère-t-il de ResNet ?**
DenseNet est une extension de ResNet où **toutes les couches** (avant pooling) sont directement connectées entre elles. Chaque couche reçoit les feature maps de toutes les couches précédentes. Avantages : flux d'information et de gradient maximaux, moins de filtres par couche.

---

**Q28. Quels sont les 3 types de segmentation d'image auxquels les CNN sont appliqués ?**
- **Détection d'objets** : localiser et classifier des objets (bounding boxes)
- **Détection de pose** : estimer les articulations du corps humain
- **Segmentation sémantique** : classer chaque pixel de l'image (ex. SegNet)

---

**Q29. Quelle est la différence entre VALID et SAME padding ?**
- **VALID :** pas de padding → la sortie est **plus petite** que l'entrée (taille réduite)
- **SAME :** padding automatique pour que la sortie ait la **même taille** que l'entrée (P = (F-1)/2)

---

**Q30. Combien de paramètres a un filtre F×F×C (sans biais) ?**
Un filtre unique : **F × F × C** paramètres.
Avec N filtres : **N × F × F × C** paramètres + **N biais**.
Exemple : 32 filtres 3×3 sur une image RGB → 32×3×3×3 + 32 = **896 paramètres** (très peu !)
