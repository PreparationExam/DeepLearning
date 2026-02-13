# Problèmes de Pratique - Chapitre 1 (SANS SOLUTIONS)

## Instructions

Ces problèmes sont pour **VOUS TESTER**. Ne regardez PAS les fichiers de solutions avant d'avoir tenté chaque problème.

**Méthode de travail :**
1. Couvrir/fermer tous les fichiers de solutions
2. Résoudre le problème complètement sur papier
3. Vérifier votre réponse dans le fichier solutions/
4. Si faux : comprendre l'erreur, refaire le problème
5. Si correct : passer au suivant

**Système de difficulté :**
-  Basique (5-7 min)
-  Standard (8-12 min)
-  Moyen (12-18 min)
-  Difficile (20-25 min)
-  Très difficile (25-30 min)

---

## Section 1 : Classification par Perceptron

### Problème 1.1 : Porte NAND

**Donné :**
- Poids : w = (-1, -1)
- Biais : w₀ = 1.5
- Activation : Heaviside
- Entrées : (0,0), (0,1), (1,0), (1,1)

**Tâches :**
a) Calculer la sortie pour chaque entrée
b) Vérifier que cela implémente la porte NAND
c) Tracer la frontière de décision

---

### Problème 1.2 : Perceptron 3D

**Donné :**
- Poids : w = (2, -1, 1)
- Biais : w₀ = -2
- Activation : Heaviside
- Entrées à classifier : (1,1,1), (0,1,1), (1,0,1), (1,1,0)

**Tâche :** 
Calculer la sortie pour chaque entrée et déterminer l'équation de l'hyperplan.

---

### Problème 1.3 : Conception inverse

**Tâche :** 
Trouver les poids (w₁, w₂) et le biais w₀ d'un perceptron qui active (sortie = 1) si et seulement si :

**2x₁ + x₂ ≥ 4**

Vérifier votre solution avec les points (0,0), (2,0), (1,3), (2,1).

---

### Problème 1.4 : Porte XOR - Preuve géométrique 
**Tâche :**
1. Dessiner les 4 points XOR sur un plan 2D avec leurs étiquettes
2. Essayer de tracer UNE ligne droite qui sépare les classes
3. Expliquer pourquoi c'est impossible
4. Que faut-il pour résoudre XOR ? (indice : combien de perceptrons ?)

---

### Problème 1.5 : Frontière de décision

**Donné :** w = (3, -2), w₀ = 6

**Tâches :**
a) Écrire l'équation de la frontière de décision
b) Trouver les intersections avec les axes
c) Tracer la ligne
d) Déterminer quelle région correspond à sortie = 1

---

### Problème 1.6 : Classification multiples points

**Donné :**
- w = (1, 1), w₀ = -2.5
- Points : A(0,0), B(1,2), C(3,0), D(2,2), E(1,1)

**Tâche :**
Classifier chaque point et dessiner la frontière de décision avec les points étiquetés.

---

## Section 2 : Fonctions d'Activation

### Problème 2.1 : Évaluation sigmoïde 

Calculer σ(x) pour x = -3, -1, 0, 2, 4

Formule : σ(x) = 1/(1 + e^(-x))

---

### Problème 2.2 : Propriété de symétrie 

**Prouver :** σ(-x) = 1 - σ(x)

Puis utiliser cette propriété pour calculer σ(-2) si σ(2) ≈ 0.881

---

### Problème 2.3 : Dérivée de sigmoïde 

**Calculer σ'(x) pour :**
a) x = 0
b) x = 1
c) x = -1
d) x = 5

Formule : σ'(x) = σ(x)(1 - σ(x))

**Quelle valeur de x donne la dérivée MAXIMALE ?**

---

### Problème 2.4 : Gradient évanescent - démonstration

**Tâche :**

Supposer un réseau à 4 couches avec activations sigmoïdes.

Si toutes les sommes pondérées z = 4 dans chaque couche :

a) Calculer σ'(4)
b) Calculer le produit des gradients : σ'(4) × σ'(4) × σ'(4) × σ'(4)
c) Expliquer pourquoi c'est un problème pour l'apprentissage
d) Comparer avec ReLU'(4)

---

### Problème 2.5 : Fonction tanh

**Donné :** tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

**Calculer :**
a) tanh(0)
b) tanh(2)
c) tanh(-2)

**Prouver :** tanh(-x) = -tanh(x)

---

### Problème 2.6 : Comparaison ReLU vs Sigmoid

**Pour x = -2, 0, 3, 10 :**

Calculer :
- ReLU(x)
- ReLU'(x)
- σ(x)
- σ'(x)

Créer un tableau comparatif et expliquer pourquoi ReLU est préféré pour les réseaux profonds.

---

### Problème 2.7 : Dying ReLU

**Scénario :**
Un neurone a : w = (-2, -3), w₀ = -5

**Pour l'entrée x = (1, 1) :**
a) Calculer z = w·x + w₀
b) Calculer ReLU(z)
c) Calculer ReLU'(z)
d) Si le gradient de la perte est ∂L/∂a = 2, quel est ∂L/∂w₁ ?
e) Expliquer pourquoi ce neurone est "mort"

---

## Section 3 : Mises à Jour des Poids

### Problème 3.1 : Règle Delta basique 

**Donné :**
- Poids actuels : w = (0.5, 0.5), w₀ = -0.2
- Taux d'apprentissage : η = 0.1
- Entrée : x = (1, 0)
- Sortie désirée : y_d = 1
- Sortie actuelle : y = 0

**Calculer les nouveaux poids.**

---

### Problème 3.2 : Erreur négative

**Donné :**
- w = (1, 1), w₀ = 0.5
- η = 0.2
- x = (1, 1)
- y_d = 0, y = 1 (perceptron prédit 1 mais devrait être 0)

**Tâches :**
a) Calculer l'erreur e
b) Calculer Δw₁, Δw₂, Δw₀
c) Calculer les nouveaux poids
d) Vérifier que le nouveau perceptron prédit correctement

---

### Problème 3.3 : Entraînement complet - Porte OR

**Initialisation :**
- w = (0, 0), w₀ = 0
- η = 0.3
- Données : table de vérité OR complète

**Tâche :**
Effectuer 1 époque complète (4 exemples). Montrer tous les calculs intermédiaires.

---

### Problème 3.4 : Impact du taux d'apprentissage

**Même scénario :**
- w = (2, 1), w₀ = -3
- x = (1, 1), y_d = 0, y = 1
- Erreur e = -1

**Calculer les nouveaux poids pour :**
a) η = 0.05
b) η = 0.5
c) η = 1.5

**Vérifier lequel fonctionne le mieux.**

---

### Problème 3.5 : Règle de Hebb

**Donné :**
- w = (0, 0), w₀ = 0
- η = 0.2
- Exemples d'entraînement :
  - x = (1, 0), y = 1
  - x = (0, 1), y = 1
  - x = (1, 1), y = 0

**Tâche :**
Appliquer la règle de Hebb : Δw = η·x·y

**Pourquoi cette règle ne peut PAS apprendre une fonction de classification spécifique ?**

---

### Problème 3.6 : Convergence multi-étapes

**Donné :**
- w = (0, 0), w₀ = 0
- η = 0.4
- Données AND : (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1

**Tâche :**
Entraîner pendant 2 époques complètes. Montrer :
- Poids après chaque mise à jour
- Vérification après chaque époque
- Convergence ou non

---

## Section 4 : Descente de Gradient

### Problème 4.1 : Descente 1D basique

**Donné :**
- Fonction de coût : C(w) = (w - 5)²
- Point initial : w₀ = 1
- Taux d'apprentissage : η = 0.3

**Effectuer 4 itérations de descente de gradient.**

---

### Problème 4.2 : Gradient 2D

**Donné :**
- C(w₁, w₂) = w₁² + 4w₂²
- Point initial : (w₁, w₂) = (2, 1)
- η = 0.2

**Tâches :**
a) Calculer ∇C = (∂C/∂w₁, ∂C/∂w₂)
b) Effectuer 2 itérations
c) Calculer C après chaque itération pour vérifier la décroissance

---

### Problème 4.3 : Détection de divergence

**Donné :**
- C(w) = w²
- w₀ = 2

**Tester 3 itérations avec :**
a) η = 0.5 (devrait converger)
b) η = 1.5 (devrait diverger)

**Identifier le seuil critique de η.**

---

### Problème 4.4 : Fonction non-convexe 
**Donné :**
- C(w) = w⁴ - 5w² + 4
- w₀ = 0.5
- η = 0.1

**Tâches :**
a) Trouver tous les points critiques (où dC/dw = 0)
b) Classifier chaque point (min/max/selle)
c) Effectuer 3 itérations depuis w₀ = 0.5
d) Vers quel minimum local converge-t-on ?

---

### Problème 4.5 : Connexion Delta-Gradient
**Donné :**
- Perceptron avec activation sigmoïde : ŷ = σ(w·x + w₀)
- Coût MSE : C = (1/2)(y - ŷ)²

**Prouver :**

∂C/∂w = -(y - ŷ) · σ'(z) · x

où z = w·x + w₀

**Ensuite montrer que la mise à jour de descente de gradient est équivalente à la règle Delta.**

---

### Problème 4.6 : Optimisation avec contrainte 

**Donné :**
- C(w₁, w₂) = w₁² + w₂²
- Contrainte : w₁ + w₂ = 4

**Tâches :**
a) Sans descente de gradient, trouver le minimum analytiquement
b) Vérifier avec 3 itérations de descente (démarrer à w₁=3, w₂=1, η=0.3)
c) Comparer les résultats

---

## Section 5 : Problèmes Intégrés

### Problème 5.1 : Pipeline complet 

**Scénario :**
Concevoir et entraîner un perceptron pour classifier :
- Classe 1 : points avec x₁ + 2x₂ > 5
- Classe 0 : sinon

**Tâches :**
a) Proposer des poids initiaux
b) Créer 6 exemples d'entraînement (3 de chaque classe)
c) Entraîner 1 époque avec η = 0.2
d) Vérifier les performances finales

---

### Problème 5.2 : Analyse d'erreur

**Donné un perceptron entraîné :**
- w = (1.2, 0.8), w₀ = -2
- Fonction cible : AND

**Tâches :**
a) Tester sur les 4 entrées AND
b) Identifier les erreurs
c) Calculer le taux d'erreur
d) Proposer une mise à jour pour corriger les erreurs

---

### Problème 5.3 : Comparaison activation

**Pour l'entrée x = (2, -1) avec w = (1, 1), w₀ = -0.5 :**

Calculer la sortie avec :
a) Heaviside
b) Sigmoid
c) Tanh
d) ReLU

**Quelle activation donne la plus grande magnitude de gradient ?**

---

### Problème 5.4 : Optimisation de η

**Expérience :**
Fonction C(w) = (w-2)², w₀ = 0

**Tester η ∈ {0.1, 0.5, 0.9, 1.1, 2.0}**

Pour chaque η :
- Effectuer 5 itérations
- Noter si convergence ou divergence
- Compter le nombre d'itérations pour atteindre |w - 2| < 0.1

**Quel est le η optimal ?**

---

### Problème 5.5 : Réseau à 2 couches (préparation Ch2)

**Réseau :**
```
Entrée (x₁, x₂) 
  → Neurone caché : h = σ(w₁x₁ + w₂x₂ + b₁)
  → Sortie : y = σ(w₃h + b₂)
```

**Donné :**
- w₁ = 1, w₂ = 1, b₁ = -1.5 (couche cachée)
- w₃ = 1, b₂ = -0.5 (couche sortie)
- Entrée : x = (1, 1)
- Sortie désirée : y_d = 0

**Tâches :**
a) Propagation avant : calculer h et y
b) Calculer l'erreur
c) Calculer ∂C/∂w₃ (gradient pour la sortie)
d) Calculer ∂C/∂w₁ (gradient pour la couche cachée - utiliser règle de chaîne)

**Ceci est un avant-goût de la rétropropagation !**

---

## Instructions Finales

**Stratégie de pratique :**

**Niveau Débutant :**
- Objectif : 80% correct

**Niveau Intermédiaire :**
- Objectif : 70% correct

**Niveau Avancé :**
- Objectif : 60% correct (ces problèmes sont difficiles!)

**Préparation Examen :**
- Faire TOUS les problèmes sans regarder les solutions
- Chronométrer vous-même
- Refaire les problèmes ratés jusqu'à 100% de réussite

**BONNE CHANCE !**