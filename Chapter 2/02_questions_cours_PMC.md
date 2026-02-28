# ❓ Questions de Cours — Perceptron Multi-Couches (PMC)

> Toutes les définitions, concepts, et questions qui peuvent tomber à l'exam.

---

## BLOC 1 — Limites du Perceptron

**Q1. Qu'est-ce qu'un perceptron simple ?**
Un modèle de base en classification supervisée binaire qui sépare deux classes à l'aide d'une frontière **linéaire** (droite ou hyperplan). Il ne fonctionne que si les données sont **linéairement séparables**.

---

**Q2. Qu'est-ce que la séparabilité linéaire ?**
Des données sont **linéairement séparables** s'il existe une droite (en 2D) ou un hyperplan (en nD) capable de séparer correctement toutes les observations de deux classes distinctes.

---

**Q3. Qui a démontré les limites du perceptron simple, et quand ?**
**Minsky et Papert en 1969** ont démontré que le perceptron simple ne peut pas résoudre certains problèmes élémentaires, notamment la fonction **XOR**.

---

**Q4. Pourquoi le perceptron simple ne peut-il pas résoudre XOR ? Démontrez-le.**
La fonction XOR n'est pas linéairement séparable. Preuve par contradiction :

Supposons ∃ (w₁, w₂, w₀) tels que XOR(x₁, x₂) = w₁x₁ + w₂x₂ + w₀.

- (0,0) : w₀ = 0
- (1,0) : w₁ + w₀ = 1
- (0,1) : w₂ + w₀ = 1
- (1,1) : w₁ + w₂ + w₀ = 0

Les 3 premières donnent w₁ + w₂ + w₀ = 2, mais la 4ème impose w₁ + w₂ + w₀ = 0. **Contradiction** → impossible. ✗

---

**Q5. Qu'est-ce que le Théorème d'Approximation Universelle ?**
Ce théorème stipule qu'un PMC avec suffisamment de couches et de neurones peut **approximer n'importe quelle fonction** de classification ou de régression, aussi complexe soit-elle.

---

## BLOC 2 — Structure du PMC

**Q6. Qu'est-ce qu'un PMC (Perceptron Multi-Couches) ?**
Un réseau de neurones artificiel composé de **plusieurs couches de perceptrons assemblées successivement** : une couche d'entrée, une ou plusieurs couches cachées, et une couche de sortie. Il permet de traiter des problèmes non linéaires.

---

**Q7. Décrivez les 3 types de couches d'un PMC.**
- **Couche d'entrée :** reçoit les variables d'entrée (features). Nombre de neurones = nombre de features.
- **Couches cachées :** couches intermédiaires qui modélisent des relations complexes. Nombre choisi par expérimentation.
- **Couche de sortie :** produit le résultat final. Nombre de neurones dépend de la tâche.

---

**Q8. Qu'est-ce qu'un réseau de neurones profond ?**
Un réseau comportant **au moins deux couches cachées**. Synonyme : Deep Neural Network (DNN).

---

**Q9. Comment choisit-on le nombre de neurones dans les couches cachées ?**
Par **expérimentation (trial & error)**. Plus il y a de neurones, plus le réseau peut apprendre des relations complexes, mais cela augmente aussi le risque de **surapprentissage**.

---

**Q10. Comment les neurones sont-ils connectés dans un PMC ?**
- Les neurones d'**une même couche ne sont PAS connectés** entre eux.
- Chaque neurone est connecté à **TOUS** les neurones de la couche précédente et de la couche suivante (architecture fully connected).

---

**Q11. Comment choisit-on le nombre de neurones en sortie selon la tâche ?**
| Tâche | Neurones sortie | Activation |
|-------|-----------------|-----------|
| Régression simple | 1 | Aucune (linéaire) |
| Régression multivariée | p (nb valeurs) | Aucune |
| Classification binaire | 1 | Sigmoïde |
| Classification multiclasse (K classes) | K | Softmax |

---

## BLOC 3 — Sorties et Fonctions d'Activation

**Q12. Quelle est la formule de sortie d'un neurone j en régression ?**
$$\hat{y}_j = \sum_{i=1}^{R} w(i,j)\hat{y}_i + w(0,j)$$
Pas de fonction d'activation (sortie linéaire).

---

**Q13. Quelle est la formule de sortie en classification binaire ? Quelle est la règle de décision ?**
$$\hat{y}_j = \sigma\left(\sum_{i=1}^{R} w(i,j)\hat{y}_i + w(0,j)\right)$$
**Règle de décision :** si ŷⱼ > 0.5 → classe C₁, sinon → classe C₂.

---

**Q14. Qu'est-ce que la fonction Softmax ? Quand l'utilise-t-on ?**
Fonction d'activation utilisée en **classification multiclasse** (K > 2). Elle transforme un vecteur de valeurs en une **distribution de probabilités** (sorties entre 0 et 1, somme = 1). La classe prédite est celle avec la probabilité maximale : ŷᵢ = arg max(ŷⱼ).

---

**Q15. Qu'est-ce que l'encodage One-Hot ?**
Représentation d'une classe par un vecteur binaire où la classe prédite prend la valeur **1** et toutes les autres prennent la valeur **0**.
Exemple : pour 3 classes {rouge, vert, bleu} :
- rouge → [1, 0, 0], vert → [0, 1, 0], bleu → [0, 0, 1]

---

## BLOC 4 — Apprentissage et Rétropropagation

**Q16. Qu'est-ce que le gradient local d'un neurone de sortie j ?**
$$\delta_j^{[k]} = \hat{y}_j^{[k]}(1 - \hat{y}_j^{[k]})\left(y_j^{d[k]} - \hat{y}_j^{[k]}\right)$$
C'est le produit de la dérivée de la sigmoïde et de l'erreur brute. Il sert à calculer les mises à jour des poids.

---

**Q17. Quelle est la formule de mise à jour des poids en couche de sortie ?**
$$\Delta W^{[k]} = \frac{\beta}{m} \delta^{[k]} (\hat{Y}^{[k-1]})^T$$
avec β le taux d'apprentissage et m le nombre d'exemples.

---

**Q18. Quelle est la formule du gradient local d'un neurone caché j ?**
$$\delta_j^{[k-1]} = \hat{y}_j^{[k-1]}(1 - \hat{y}_j^{[k-1]}) \sum_{r \in \text{dest}(j)} w^{[k]}(j,r) \delta_r^{[k]}$$
L'erreur se **propage en arrière** depuis les neurones de la couche suivante.

---

**Q19. Qu'est-ce que la rétropropagation (backpropagation) ?**
Algorithme d'apprentissage du PMC qui :
1. Calcule les δⱼ pour les neurones de **sortie**
2. Propage progressivement ces erreurs vers les **couches cachées** (de la dernière vers la première)
3. Met à jour les **poids** pour réduire l'écart entre la sortie prédite et la valeur réelle

---

**Q20. Qu'est-ce que la saturation d'un neurone ? Pourquoi est-ce problématique ?**
Un neurone sigmoïde est **saturé** quand sa sortie est proche de 0 ou de 1 (entrée très grande ou très petite en valeur absolue).
Conséquence : le **gradient est quasi nul** → les poids ne se mettent presque plus à jour → **apprentissage très lent**.

---

**Q21. Comment éviter la saturation des neurones ?**
**Normaliser les données à l'entrée du PMC :** transformer chaque variable xᵢ dans l'intervalle [-1/max(xᵢ), 1/max(xᵢ)], c'est-à-dire dans [-1, 1].

---

**Q22. Pourquoi transforme-t-on les valeurs désirées en {0.05, 0.95} plutôt que {0, 1} ?**
Les valeurs 0 et 1 correspondent à des sorties sigmoïde qui nécessitent des activations infinies (±∞), ce qui cause une saturation. En utilisant **0.05 et 0.95**, on reste dans la plage opératoire de la sigmoïde et on évite ce problème.

---

**Q23. Comment initialise-t-on les poids d'un PMC ?**
Aléatoirement, selon une **distribution uniforme dans [-0.5, 0.5]** avant le début de l'apprentissage. Cette initialisation aléatoire rend le PMC un **algorithme stochastique**.

---

**Q24. Qu'est-ce que le taux d'apprentissage β ? Comment le choisir ?**
β contrôle l'**amplitude des mises à jour des poids** lors de la descente du gradient. Sa valeur est entre 0 et 1. Il n'existe pas de règle universelle, le choix est **empirique**.

Stratégies :
- **Diminution progressive** : grand β au début, puis réduction
- **Alternance** : alterner petite/grande valeur selon les phases

---

**Q25. Décrivez le déroulement complet de l'algorithme PMC.**
1. Normaliser les données : xᵢ ∈ [-1, 1], yᵈ ∈ {0.05, 0.95}
2. Initialiser les poids : w(i,j) ∈ [-0.5, 0.5]
3. Répéter jusqu'au critère d'arrêt :
   a. Choisir aléatoirement un exemple (Xᵢ, yᵢᵈ)
   b. Calculer δⱼ et Δw(i,j) pour les neurones de sortie (couche de sortie)
   c. Calculer δⱼ et Δw(i,j) pour les neurones cachés (couches cachées)
   d. Mettre à jour : w(i,j) ← w(i,j) + Δw(i,j)

---

## BLOC 5 — Surapprentissage et Régularisation

**Q26. Qu'est-ce que la généralisation ?**
La capacité d'un modèle à obtenir de **bons résultats sur des données nouvelles, non vues lors de l'entraînement**.

---

**Q27. Quelle est la différence entre sous-apprentissage et surapprentissage ?**
- **Sous-apprentissage (underfitting) :** erreur d'entraînement élevée → le modèle est trop simple, il n'apprend pas.
- **Surapprentissage (overfitting) :** faible erreur d'entraînement mais forte erreur de test → le modèle mémorise les données d'entraînement au lieu d'apprendre des patterns généraux.

---

**Q28. Qu'est-ce que la régularisation ? Quels sont ses 3 axes ?**
La régularisation regroupe les méthodes pour **réduire le surapprentissage**. Trois axes :
1. **Modifier la fonction d'erreur** (L1, L2)
2. **Modifier les données** (data augmentation)
3. **Modifier le modèle** (dropout)

---

**Q29. Qu'est-ce que la régularisation L1 (Lasso) ? Et L2 (Ridge) ? Différences ?**
- **L1 (Lasso) :** Ω(w) = λ Σ|wᵢ| → force certains poids à devenir **exactement zéro** (sélection de features)
- **L2 (Ridge) :** Ω(w) = λ Σwᵢ² → pousse les poids vers de **petites valeurs** sans les annuler
- Le paramètre λ contrôle l'intensité : λ=0 → pas de régularisation ; grand λ → forte régularisation

---

**Q30. Qu'est-ce que le Dropout ? Comment fonctionne-t-il ?**
Technique de régularisation qui **désactive aléatoirement un pourcentage de neurones** (ex: 50%) pendant l'entraînement.
- Objectif : améliorer la généralisation, éviter que le réseau dépende de quelques neurones seulement
- Appliqué généralement dans les couches entièrement connectées vers la fin du réseau
- **Désactivé lors de l'inférence** (test)

---

**Q31. Qu'est-ce que l'augmentation de données ?**
Technique qui consiste à **créer artificiellement de nouvelles données** à partir des données existantes pour enrichir le jeu d'entraînement.
Pour les images : translation, rotation, changement d'échelle, symétrie horizontale/verticale, recadrage aléatoire.

---

**Q32. Qu'est-ce que l'opérateur ⊙ dans les formules de rétropropagation ?**
Le symbole ⊙ représente le **produit de Hadamard** (multiplication élément par élément) entre deux vecteurs ou matrices de même taille. Différent du produit matriciel classique.
