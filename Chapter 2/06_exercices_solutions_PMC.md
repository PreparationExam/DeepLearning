# 💪 50 Exercices & Problèmes avec Solutions — PMC

---

## PARTIE 1 — QCM et Questions Conceptuelles (1-15)

---

**Ex 1.** Le perceptron simple peut-il résoudre la fonction NAND (NON-ET) ?
- a) Oui  
- b) Non

**✅ Solution : a) Oui.**
NAND est linéairement séparable. Table de vérité NAND : (0,0)→1, (0,1)→1, (1,0)→1, (1,1)→0. On peut trouver une droite séparant (1,1) des trois autres points.

---

**Ex 2.** Dans un PMC, les neurones d'une même couche cachée sont :
- a) Tous connectés entre eux
- b) Partiellement connectés
- c) Non connectés entre eux

**✅ Solution : c) Non connectés entre eux.**
Les connexions existent uniquement entre couches adjacentes.

---

**Ex 3.** Quelle fonction d'activation utilise-t-on en sortie pour une classification avec 7 classes ?
- a) Sigmoïde
- b) Softmax
- c) Linéaire
- d) ReLU

**✅ Solution : b) Softmax.** La softmax est utilisée pour toute classification multiclasse (K > 2). Elle donne une distribution de probabilité sur les K classes.

---

**Ex 4.** Un réseau prédit les coordonnées (x, y, z) d'un point dans l'espace. Combien de neurones en sortie ?
- a) 1
- b) 2
- c) 3
- d) 9

**✅ Solution : c) 3.** Régression multivariée : 1 neurone par valeur à prédire → x, y, z = 3 neurones. Activation : linéaire.

---

**Ex 5.** Vrai ou Faux : Le PMC est un algorithme déterministe.

**✅ Solution : FAUX.** Le PMC est un algorithme **stochastique** car les poids sont initialisés aléatoirement. Deux entraînements successifs peuvent donner des résultats légèrement différents.

---

**Ex 6.** Quelle est la plage de valeurs de la fonction sigmoïde ?
- a) ]-∞, +∞[
- b) [0, 1]
- c) [-1, 1]
- d) [0, +∞[

**✅ Solution : b) [0, 1].** σ(x) = 1/(1+e^{-x}), sa valeur est toujours entre 0 et 1 (strictement).

---

**Ex 7.** Un neurone sigmoïde est dit "saturé" quand :
- a) Sa sortie est proche de 0.5
- b) Sa sortie est proche de 0 ou 1
- c) Son gradient est très élevé
- d) Ses poids sont trop grands

**✅ Solution : b) Sa sortie est proche de 0 ou 1.** Dans ce cas, le gradient est quasi nul → l'apprentissage est bloqué.

---

**Ex 8.** Pourquoi utilise-t-on {0.05, 0.95} comme valeurs désirées plutôt que {0, 1} ?

**✅ Solution :** Les valeurs 0 et 1 sont les **asymptotes** de la sigmoïde : y=0 nécessite une activation → -∞ et y=1 nécessite → +∞. Cela provoquerait une saturation des neurones de sortie et bloquerait l'apprentissage. Les valeurs 0.05 et 0.95 restent dans la plage opératoire de la sigmoïde.

---

**Ex 9.** Vrai ou Faux : Augmenter le nombre de couches cachées améliore toujours les performances.

**✅ Solution : FAUX.** Trop de couches ou de neurones augmente le risque de **surapprentissage (overfitting)**. Il faut trouver le bon compromis biais-variance.

---

**Ex 10.** Quel est l'effet d'un taux d'apprentissage β trop élevé ?
- a) Apprentissage très lent
- b) Oscillations, risque de divergence
- c) Surapprentissage
- d) Saturation des neurones

**✅ Solution : b) Oscillations, risque de divergence.** Le réseau "saute" par-dessus le minimum de la fonction de coût.

---

**Ex 11.** Pour encoder la classe "chat" dans un système de classification à 4 classes {chien, chat, oiseau, poisson}, quel est le vecteur one-hot ?

**✅ Solution :** [0, 1, 0, 0]  
La classe "chat" est en position 2 (index 1) → 1 à la position 1, 0 partout ailleurs.

---

**Ex 12.** Quelle est la différence entre erreur d'apprentissage et erreur de généralisation ?

**✅ Solution :**
- **Erreur d'apprentissage** : calculée sur les données d'entraînement (données connues du modèle)
- **Erreur de généralisation** : calculée sur des données nouvelles, non vues lors de l'entraînement → mesure la vraie performance du modèle

---

**Ex 13.** Décrivez le phénomène d'overfitting en une phrase.

**✅ Solution :** Le modèle **mémorise** les données d'entraînement (y compris le bruit) au lieu d'apprendre des motifs généraux → faible erreur d'entraînement mais forte erreur sur données nouvelles.

---

**Ex 14.** Quelle régularisation tendrait à annuler complètement certains poids : L1 ou L2 ?

**✅ Solution : L1 (Lasso).** La pénalisation |w| crée un point anguleux en 0 qui favorise des solutions sparse (des poids exactement nuls). L2 réduit les poids mais ne les annule pas.

---

**Ex 15.** Vrai ou Faux : Le dropout est appliqué lors de la phase de test/inférence.

**✅ Solution : FAUX.** Le dropout est désactivé lors de l'inférence. Tous les neurones sont actifs en test. Seul l'entraînement utilise le dropout.

---

## PARTIE 2 — Calculs Numériques (16-30)

---

**Ex 16.** Calculez σ(0), σ(1), σ(-1), σ(2), σ(-2).

**✅ Solution :**
- σ(0) = 1/(1+e⁰) = 1/2 = **0.5**
- σ(1) = 1/(1+e⁻¹) = 1/(1+0.368) ≈ **0.731**
- σ(-1) = 1/(1+e¹) = 1/(1+2.718) ≈ **0.269**
- σ(2) = 1/(1+e⁻²) ≈ 1/(1+0.135) ≈ **0.880**
- σ(-2) = 1/(1+e²) ≈ 1/(1+7.389) ≈ **0.119**

---

**Ex 17.** Calculez la dérivée σ'(x) = σ(x)(1-σ(x)) pour x = 0 et x = 2.

**✅ Solution :**
- x=0 : σ(0)=0.5, σ'(0) = 0.5 × 0.5 = **0.25** (valeur max)
- x=2 : σ(2)≈0.88, σ'(2) = 0.88 × 0.12 ≈ **0.106**

---

**Ex 18.** Un neurone a les entrées x₁=2, x₂=-1, les poids w₁=0.5, w₂=0.3, et le biais w₀=-0.2.
Calculez sa sortie avec activation sigmoïde.

**✅ Solution :**
```
S = w₁·x₁ + w₂·x₂ + w₀
S = 0.5×2 + 0.3×(-1) + (-0.2)
S = 1 - 0.3 - 0.2 = 0.5
ŷ = σ(0.5) = 1/(1+e^{-0.5}) = 1/(1+0.607) ≈ 0.622
```

---

**Ex 19.** Pour le neurone de l'ex. 18, si la valeur désirée est yᵈ = 0.95, calculez le gradient local δ.

**✅ Solution :**
```
δ = ŷ(1-ŷ)(yᵈ - ŷ)
δ = 0.622 × (1-0.622) × (0.95 - 0.622)
δ = 0.622 × 0.378 × 0.328
δ ≈ 0.077
```

---

**Ex 20.** Avec δ = 0.077 (ex. 19), β = 0.1, calculez les mises à jour des poids Δw₁ et Δw₂.

**✅ Solution :**
```
Δw(i,j) = β × δ × entrée_i
Δw₁ = 0.1 × 0.077 × 2 = 0.0154
Δw₂ = 0.1 × 0.077 × (-1) = -0.0077
Δw₀ = 0.1 × 0.077 × 1 = 0.0077  (biais, entrée = 1)
```

---

**Ex 21.** Un réseau a 3 entrées, 1 couche cachée de 4 neurones, et 2 sorties. Combien de poids (sans compter les biais) ?

**✅ Solution :**
- Connexions entrée→cachée : 3 × 4 = 12
- Connexions cachée→sortie : 4 × 2 = 8
- **Total poids : 12 + 8 = 20**

Biais : 4 (cachée) + 2 (sortie) = 6
Total paramètres : **26**

---

**Ex 22.** Calculez le nombre total de paramètres (poids + biais) d'un PMC avec couches [784, 128, 64, 10].

**✅ Solution :**
- 784→128 : 784×128 + 128 = 100 352 + 128 = 100 480
- 128→64 : 128×64 + 64 = 8 192 + 64 = 8 256
- 64→10 : 64×10 + 10 = 640 + 10 = 650
- **Total : 100 480 + 8 256 + 650 = 109 386 paramètres**

---

**Ex 23.** Propagation avant dans un petit PMC.

Réseau 2 entrées → 1 neurone caché → 1 sortie.
- x₁ = 1, x₂ = 0
- Poids couche cachée (neurone 3) : w(0,3)=0, w(1,3)=1, w(2,3)=0.5
- Poids couche sortie (neurone 4) : w(0,4)=-0.5, w(3,4)=2
- Activation : sigmoïde partout

**✅ Solution :**
```
Étape 1 — Neurone caché 3 :
S₃ = 0 + 1×1 + 0.5×0 = 1
ŷ₃ = σ(1) ≈ 0.731

Étape 2 — Neurone sortie 4 :
S₄ = -0.5 + 2×0.731 = -0.5 + 1.462 = 0.962
ŷ₄ = σ(0.962) = 1/(1+e^{-0.962}) ≈ 0.724
```

---

**Ex 24.** Rétropropagation pour l'ex. 23, avec yᵈ = 0.05 et β = 1.

**✅ Solution :**
```
δ₄ (sortie) = ŷ₄(1-ŷ₄)(yᵈ-ŷ₄)
            = 0.724 × 0.276 × (0.05 - 0.724)
            = 0.724 × 0.276 × (-0.674)
            ≈ -0.135

ΔW(3,4) = β × δ₄ × ŷ₃ = 1 × (-0.135) × 0.731 ≈ -0.099
ΔW(0,4) = β × δ₄ × 1  = -0.135

δ₃ (caché) = ŷ₃(1-ŷ₃) × w(3,4) × δ₄
           = 0.731 × 0.269 × 2 × (-0.135)
           ≈ -0.053

ΔW(1,3) = β × δ₃ × x₁ = (-0.053) × 1 = -0.053
ΔW(2,3) = β × δ₃ × x₂ = (-0.053) × 0 = 0
ΔW(0,3) = β × δ₃ × 1  = -0.053
```

---

**Ex 25.** Normalisation : les valeurs d'une variable sont {10, 20, 50, 100}. Normalisez-les dans [-1, 1].

**✅ Solution :**
max(xᵢ) = 100, donc on divise par 100 :
- 10 → 10/100 = **0.10**
- 20 → 20/100 = **0.20**
- 50 → 50/100 = **0.50**
- 100 → 100/100 = **1.00**

(Ou utiliser [-1/max, 1/max] × x si données négatives)

---

**Ex 26.** Calculez σ(70.5) et expliquez ce que cela illustre.

**✅ Solution :**
σ(70.5) = 1/(1+e^{-70.5}) ≈ 1/(1+0) ≈ **1**

Cela illustre la **saturation** : pour une entrée aussi grande, la sigmoïde est complètement saturée à 1. Le gradient δ = ŷ(1-ŷ) ≈ 1 × 0 = 0 → aucun apprentissage possible.

---

**Ex 27.** Softmax sur le vecteur [2.0, 1.0, 0.1] — calculez les probabilités.

**✅ Solution :**
```
e^2.0 = 7.389,  e^1.0 = 2.718,  e^0.1 = 1.105
Somme = 7.389 + 2.718 + 1.105 = 11.212

P₁ = 7.389/11.212 ≈ 0.659
P₂ = 2.718/11.212 ≈ 0.242
P₃ = 1.105/11.212 ≈ 0.099
```
Classe prédite : **classe 1** (probabilité max 0.659)
Vérification : 0.659 + 0.242 + 0.099 ≈ 1 ✓

---

**Ex 28.** Calculez la régularisation L2 pour les poids w = [0.5, -0.3, 0.8, -0.1] avec λ = 0.01.

**✅ Solution :**
```
Ω(w) = λ × Σwᵢ²
     = 0.01 × (0.5² + (-0.3)² + 0.8² + (-0.1)²)
     = 0.01 × (0.25 + 0.09 + 0.64 + 0.01)
     = 0.01 × 0.99
     = 0.0099
```

---

**Ex 29.** Calculez la régularisation L1 pour les mêmes poids avec λ = 0.01.

**✅ Solution :**
```
Ω(w) = λ × Σ|wᵢ|
     = 0.01 × (|0.5| + |-0.3| + |0.8| + |-0.1|)
     = 0.01 × (0.5 + 0.3 + 0.8 + 0.1)
     = 0.01 × 1.7
     = 0.017
```

---

**Ex 30.** Mise à jour complète d'un poids avec L2. Poids w = 0.5, gradient δ = 0.1, x = 1, β = 0.1, λ = 0.01, m = 1.

**✅ Solution :**
Avec L2, la mise à jour devient :
```
Δw = β/m × δ × x - β × λ × w    (terme de régularisation)
   = 0.1 × 0.1 × 1 - 0.1 × 0.01 × 0.5
   = 0.01 - 0.0005
   = 0.0095

w_new = 0.5 + 0.0095 = 0.5095
```

---

## PARTIE 3 — Problèmes Approfondis (31-45)

---

**Ex 31.** Un ingénieur entraîne un PMC pour classer des e-mails (spam/ham). Après entraînement :
- Erreur train : 1%
- Erreur test : 35%
Que se passe-t-il ? Quelles solutions proposez-vous ?

**✅ Solution :**
C'est un **surapprentissage sévère** (overfitting). Le modèle a mémorisé les données d'entraînement.

Solutions :
1. Ajouter de la **régularisation L2** à la fonction d'erreur
2. Appliquer du **dropout** dans les couches cachées (ex: 50%)
3. **Réduire** le nombre de neurones/couches
4. Collecter plus de **données d'entraînement**
5. Utiliser de la **data augmentation** si applicable

---

**Ex 32.** Vous concevez un PMC pour reconnaître des chiffres manuscrits (MNIST : images 28×28 en niveaux de gris, 10 classes). Décrivez l'architecture complète.

**✅ Solution :**
- **Couche d'entrée :** 28×28 = **784 neurones**
- **Couches cachées :** à choisir par expérimentation, ex: [128, 64] (2 couches cachées)
- **Couche de sortie :** **10 neurones** (chiffres 0-9), activation **Softmax**
- Activation cachées : sigmoïde (ou ReLU)
- Classe prédite : argmax(ŷ) → chiffre avec probabilité max
- One-hot : ex. chiffre "3" → [0,0,0,1,0,0,0,0,0,0]

---

**Ex 33.** Expliquez pourquoi l'initialisation à zéro de tous les poids est problématique.

**✅ Solution :**
Si tous les poids sont initialisés à 0, tous les neurones d'une même couche calculent exactement la même somme pondérée et ont le même gradient. Ils évoluent de façon identique pendant toute l'entraînement → **symétrie brisante** : les neurones restent toujours identiques et le réseau n'a pas plus de capacité qu'un seul neurone. C'est pourquoi on utilise une initialisation **aléatoire**.

---

**Ex 34.** Propagation avant complète — XOR réseau.

Réseau 2 entrées, 2 neurones cachés (3,4), 1 neurone sortie (5).

Poids :
- w(0,3)=0.2, w(1,3)=0.1, w(2,3)=0.3
- w(0,4)=-0.3, w(1,4)=-0.2, w(2,4)=0.4
- w(0,5)=0.4, w(3,5)=0.5, w(4,5)=-0.4

Entrée : x₁=0, x₂=1, yᵈ=0.95

**✅ Solution :**
```
Neurone 3 :
S₃ = 0.2 + 0.1×0 + 0.3×1 = 0.5
ŷ₃ = σ(0.5) ≈ 0.622

Neurone 4 :
S₄ = -0.3 + (-0.2)×0 + 0.4×1 = 0.1
ŷ₄ = σ(0.1) ≈ 0.525

Neurone 5 :
S₅ = 0.4 + 0.5×0.622 + (-0.4)×0.525
   = 0.4 + 0.311 - 0.21 = 0.501
ŷ₅ = σ(0.501) ≈ 0.623

Erreur brute = yᵈ - ŷ₅ = 0.95 - 0.623 = 0.327
```

---

**Ex 35.** Suite ex. 34 — Rétropropagation, β=1.

**✅ Solution :**
```
δ₅ = ŷ₅(1-ŷ₅)(yᵈ-ŷ₅)
   = 0.623 × 0.377 × 0.327 ≈ 0.077

Δw(3,5) = 1 × 0.077 × ŷ₃ = 0.077 × 0.622 ≈ 0.048
Δw(4,5) = 1 × 0.077 × ŷ₄ = 0.077 × 0.525 ≈ 0.040
Δw(0,5) = 1 × 0.077 × 1  = 0.077

δ₄ = ŷ₄(1-ŷ₄) × w(4,5) × δ₅
   = 0.525×0.475 × (-0.4) × 0.077 ≈ -0.008

δ₃ = ŷ₃(1-ŷ₃) × w(3,5) × δ₅
   = 0.622×0.378 × 0.5 × 0.077 ≈ 0.009
```

---

**Ex 36.** On entraîne un modèle pendant 1000 époques. L'erreur de train descend régulièrement mais l'erreur de test recommence à monter à partir de l'époque 300. Que faire ?

**✅ Solution :**
C'est de l'**overfitting** à partir de l'époque 300. Stratégies :
1. **Early stopping** : arrêter l'entraînement à l'époque 300 (minimum de l'erreur test)
2. Ajouter du **dropout**
3. Ajouter une **régularisation L2**
4. Réduire la complexité du réseau

---

**Ex 37.** Un PMC a 2 couches cachées de 50 neurones et est entraîné sur 100 exemples. Commentez.

**✅ Solution :**
Nombre de paramètres (supposons 10 entrées, 1 sortie) :
- 10→50 : 550, 50→50 : 2550, 50→1 : 51
- Total ≈ 3151 paramètres pour 100 exemples

C'est **trop** → risque élevé de surapprentissage car le modèle a plus de paramètres que d'exemples. Il faut soit réduire la complexité, soit collecter plus de données, soit regulariser fortement.

---

**Ex 38.** Décrivez l'ordre exact des opérations dans une itération de l'algorithme PMC.

**✅ Solution :**
1. Sélectionner aléatoirement un exemple (Xᵢ, yᵢᵈ)
2. **Propagation avant** : calculer toutes les sorties couche par couche (entrée → sortie)
3. Calculer les **gradients locaux δ de la couche de sortie**
4. **Rétropropagation** : calculer les δ des couches cachées (sortie → entrée)
5. **Mise à jour** de tous les poids w(i,j) ← w(i,j) + Δw(i,j)
6. Recommencer jusqu'au critère d'arrêt

---

**Ex 39.** Quelle est la sortie d'un softmax pour le vecteur [1, 1, 1] ?

**✅ Solution :**
```
e¹ = e¹ = e¹ = e ≈ 2.718
Somme = 3e
P₁ = P₂ = P₃ = e/(3e) = 1/3 ≈ 0.333
```
Résultat : **distribution uniforme [1/3, 1/3, 1/3]** → le réseau est incertain, aucune classe ne domine.

---

**Ex 40.** Régression : un PMC prédit la température (°C) à partir de 5 mesures météo. Concevez la couche de sortie.

**✅ Solution :**
- **1 seul neurone** en sortie (régression simple)
- **Aucune fonction d'activation** (sortie linéaire, peut prendre n'importe quelle valeur réelle)
- Formule : ŷ = Σ w(i,j)·ŷᵢ + w(0,j)

---

**Ex 41.** Pourquoi la rétropropagation se fait-elle de la sortie vers l'entrée (et non dans l'autre sens) ?

**✅ Solution :**
Car on calcule les gradients locaux δ en utilisant les gradients de la **couche suivante** (plus proche de la sortie). Pour calculer δ d'un neurone caché, on a besoin des δ des neurones vers lesquels il se projette (dest(j)). On doit donc commencer par calculer les δ en sortie, puis les propager en arrière. C'est la **règle de chaîne** (chain rule) du calcul différentiel.

---

**Ex 42.** Un réseau classify des images de fruits en 5 catégories. La sortie softmax donne : [0.05, 0.70, 0.10, 0.10, 0.05]. Quelle est la classe prédite ? Quel est le one-hot correspondant ?

**✅ Solution :**
- Classe prédite : **classe 2** (index 1, probabilité 0.70 = max)
- Vecteur one-hot : **[0, 1, 0, 0, 0]**
- Le modèle est assez confiant (70%) mais pas certain à 100%

---

**Ex 43.** Calculez manuellement σ(0.6) et σ(-0.1) comme dans l'exemple XOR du cours.

**✅ Solution :**
```
σ(0.6) = 1/(1+e^{-0.6}) = 1/(1+0.5488) = 1/1.5488 ≈ 0.6457 ≈ 0.65 ✓

σ(-0.1) = 1/(1+e^{0.1}) = 1/(1+1.1052) = 1/2.1052 ≈ 0.4750 ≈ 0.48 ✓
```
Ces valeurs correspondent exactement à l'exemple XOR du cours pour (x₁=1, x₂=1).

---

**Ex 44.** Un PMC a été entraîné et donne : train=5%, test=6%. Puis on l'entraîne plus longtemps : train=2%, test=15%. Que s'est-il passé ?

**✅ Solution :**
Entre les deux états :
- **Avant :** bon équilibre, légère différence train/test → bonne généralisation
- **Après :** sur-apprentissage clair → en continuant l'entraînement, le modèle s'est mis à mémoriser les données
- Solution : utiliser un **critère d'arrêt** basé sur l'erreur de validation (early stopping), ou de la régularisation

---

**Ex 45.** Démontrez que la dérivée de σ(x) = 1/(1+e^{-x}) est σ(x)(1-σ(x)).

**✅ Solution :**
```
σ(x) = 1/(1+e^{-x}) = (1+e^{-x})^{-1}

dσ/dx = -(1+e^{-x})^{-2} × (-e^{-x})
      = e^{-x} / (1+e^{-x})²
      = [1/(1+e^{-x})] × [e^{-x}/(1+e^{-x})]
      = σ(x) × [(1+e^{-x}-1)/(1+e^{-x})]
      = σ(x) × [1 - 1/(1+e^{-x})]
      = σ(x) × (1-σ(x))  ✓
```

---

## PARTIE 4 — Questions de Synthèse (46-50)

---

**Ex 46.** Comparez le perceptron simple et le PMC sur 5 critères.

**✅ Solution :**

| Critère | Perceptron Simple | PMC |
|---------|------------------|-----|
| Frontière de décision | Linéaire seulement | Non linéaire |
| Problèmes résolus | AND, OR, NAND | XOR, spirales, etc. |
| Architecture | 1 seule couche | Plusieurs couches |
| Apprentissage | Règle delta simple | Rétropropagation |
| Puissance | Limitée | Approximation universelle |

---

**Ex 47.** Un étudiant propose d'initialiser tous les poids à 0.5 (constante, pas aléatoire). Quels problèmes va-t-il rencontrer ?

**✅ Solution :**
Si tous les poids d'une couche sont identiques :
1. Tous les neurones de cette couche produiront la **même sortie** → symmétrie
2. Leurs gradients δ seront **identiques**
3. Leurs poids se mettront à jour **de la même façon**
4. Résultat : les neurones restent toujours identiques → la couche se comporte comme si elle n'avait **qu'un seul neurone** → capacité très réduite
5. Solution : initialisation **aléatoire** brise cette symétrie

---

**Ex 48.** Expliquez pourquoi le dropout améliore la généralisation.

**✅ Solution :**
Le dropout force le réseau à **ne pas dépendre d'un sous-ensemble fixe de neurones** :
1. À chaque itération, un sous-ensemble différent de neurones est actif
2. Le réseau doit apprendre des **représentations redondantes** et robustes
3. C'est équivalent à entraîner un **ensemble de modèles différents** et à en faire la moyenne lors du test
4. Le réseau devient moins sensible au bruit et aux exemples individuels → meilleure généralisation

---

**Ex 49.** On vous donne deux réseaux :
- Réseau A : [input, 200 neurones cachés, output]
- Réseau B : [input, 50, 50, 50, 50 neurones cachés, output]

Avec le même nombre de paramètres total, lequel choisir pour un problème complexe ? Pourquoi ?

**✅ Solution :**
Préférer le **Réseau B** (plus profond mais moins large) :
1. Les réseaux **profonds** apprennent des **représentations hiérarchiques** (features simples → complexes)
2. Plus de couches = plus de transformations non linéaires empilées
3. En pratique, la profondeur est souvent plus efficace que la largeur pour les problèmes complexes
4. C'est le principe des **réseaux de neurones profonds (deep learning)**

---

**Ex 50.** Cas pratique complet : vous entraînez un PMC pour détecter les fraudes bancaires (données déséquilibrées : 99% transactions normales, 1% fraudes). Quels problèmes anticiper et quelles solutions proposer ?

**✅ Solution :**
**Problèmes :**
1. **Déséquilibre de classes** : le modèle peut prédire "toujours normal" et avoir 99% de précision en train mais être inutile
2. **Surapprentissage** sur la classe majoritaire
3. **Métrique inadaptée** : l'accuracy seule est trompeuse

**Solutions :**
1. Utiliser des métriques adaptées : **précision, rappel, F1-score, AUC-ROC**
2. **Rééchantillonnage** : sous-sampler la classe majoritaire OU sur-sampler la minoritaire (SMOTE)
3. **Pondération des classes** : donner plus de poids aux exemples de fraude dans la fonction d'erreur
4. Régularisation pour éviter le surapprentissage
5. Valeurs désirées : utiliser **0.05/0.95** pour éviter la saturation
6. Normaliser toutes les features en [-1, 1]
