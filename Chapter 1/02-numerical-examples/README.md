# README - Chapitre 1 : Réseaux de Neurones Fondamentaux

## Vue d'ensemble

Ce dossier contient **tout ce dont tu as besoin** pour maîtriser le Chapitre 1 de Deep Learning.

**Contenu :**
- 21 problèmes résolus avec solutions complètes
- 30 problèmes de pratique sans solutions
- Antisèches de formules
- Stratégies de résolution étape par étape

---

## Comment utiliser ce dossier

### Méthode d'apprentissage recommandée

**ÉTAPE 1 :**
1. Lire `01-perceptron-classification/classification-problems.md` - TOUTES les solutions
2. Lire `02-activation-functions/activation-functions.md` - TOUTES les solutions
3. Lire `03-weight-updates/weight-updates.md` - TOUTES les solutions
4. Lire `04-gradient-descent/gradient-descent.md` - TOUTES les solutions

**ÉTAPE 2 :**
1. Ouvrir `practice-problems.md`
2. Résoudre CHAQUE problème sur papier (sans regarder les solutions)
3. Vérifier tes réponses dans les fichiers de solutions
4. Refaire les problèmes ratés

**ÉTAPE 3 :**
1. Refaire TOUS les problèmes sans notes
2. Te chronométrer pour simuler l'examen
3. Créer tes propres variations de problèmes

---

## ANTISÈCHE FORMULES - MÉMORISE CECI

### 1. Perceptron de base

**Équation du perceptron :**
```
S = Σ(wᵢxᵢ) + w₀
y = f(S)
```

**Où :**
- `S` = somme pondérée
- `wᵢ` = poids synaptique de l'entrée i
- `xᵢ` = entrée i
- `w₀` = biais
- `f` = fonction d'activation

**Frontière de décision (hyperplan) :**
```
Σ(wᵢxᵢ) + w₀ = 0
```

Pour 2D :
```
w₁x₁ + w₂x₂ + w₀ = 0
→ x₂ = -(w₁/w₂)x₁ - (w₀/w₂)
```

---

### 2. Fonctions d'activation

**Fonction de Heaviside (seuil) :**
```
H(x) = { 1  si x ≥ 0
       { 0  si x < 0
```

**Fonction sigmoïde :**
```
σ(x) = 1 / (1 + e^(-x))

Propriétés :
- Range : (0, 1)
- σ(0) = 0.5
- σ(-x) = 1 - σ(x)
- σ'(x) = σ(x)(1 - σ(x))
- σ'(0) = 0.25 (maximum)
```

**Fonction tangente hyperbolique :**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Propriétés :
- Range : (-1, 1)
- tanh(0) = 0
- tanh(-x) = -tanh(x)
- tanh'(x) = 1 - tanh²(x)
```

**Fonction ReLU (Rectified Linear Unit) :**
```
ReLU(x) = max(0, x) = { x  si x > 0
                      { 0  si x ≤ 0

ReLU'(x) = { 1  si x > 0
           { 0  si x ≤ 0
```

---

### 3. Règles d'apprentissage

**Règle Delta (apprentissage supervisé) :**
```
e = y_désiré - y_actuel

Δwᵢ = η · e · xᵢ
Δw₀ = η · e · 1

w_nouveau = w_ancien + Δw
```

**Où :**
- `e` = erreur
- `η` = taux d'apprentissage (learning rate)
- `xᵢ` = valeur de l'entrée i

**Règle de Hebb (apprentissage non-supervisé) :**
```
Δwᵢ = η · xᵢ · y

(PAS d'erreur - seulement corrélation!)
```

---

### 4. Descente de gradient

**Formule générale :**
```
w_nouveau = w_ancien - η · ∇C(w)
```

**Où :**
- `∇C(w)` = gradient de la fonction de coût
- Pour 1D : `∇C = dC/dw`
- Pour 2D : `∇C = (∂C/∂w₁, ∂C/∂w₂)`

**Gradient pour perceptron (MSE) :**
```
C = (1/2)(y - ŷ)²

∂C/∂wᵢ = -(y - ŷ) · f'(S) · xᵢ
```

---

## GUIDE DE RÉSOLUTION ÉTAPE PAR ÉTAPE

### Type 1 : Classification par perceptron

**MÉTHODE :**

**Étape 1 :** Écrire la formule
```
S = w₁x₁ + w₂x₂ + ... + wₙxₙ + w₀
```

**Étape 2 :** Calculer S pour chaque entrée
```
Exemple : w=(1,1), w₀=-1.5, x=(1,0)
S = 1·1 + 1·0 + (-1.5) = -0.5
```

**Étape 3 :** Appliquer la fonction d'activation
```
Si Heaviside : y = 1 si S ≥ 0, sinon y = 0
Donc : S = -0.5 < 0 → y = 0
```

**Étape 4 :** Vérifier/dessiner la frontière
```
Équation : w₁x₁ + w₂x₂ + w₀ = 0
```

**PIÈGES À ÉVITER :**
- Oublier le biais w₀
- Mauvais signe sur w₀
- Confondre S ≥ 0 avec S > 0

---

### Type 2 : Évaluation fonction d'activation

**MÉTHODE :**

**Pour σ(x) :**

**Étape 1 :** Utiliser la formule
```
σ(x) = 1/(1 + e^(-x))
```

**Étape 2 :** Cas spéciaux à mémoriser
```
σ(0) = 0.5 (toujours!)
σ(∞) = 1
σ(-∞) = 0
```

**Étape 3 :** Utiliser la symétrie
```
σ(-x) = 1 - σ(x)
Donc si σ(2) = 0.88, alors σ(-2) = 0.12
```

**Pour la dérivée :**
```
σ'(x) = σ(x)(1 - σ(x))

Exemple : σ(0) = 0.5
σ'(0) = 0.5 · (1 - 0.5) = 0.25
```

**PIÈGES À ÉVITER :**
- Dire σ(0) = 0 (c'est 0.5!)
- Oublier d'utiliser σ(x) déjà calculé pour σ'(x)
- Confondre σ(x) et σ'(x)

---

### Type 3 : Mise à jour des poids (Règle Delta)

**MÉTHODE :**

**Étape 1 :** Calculer l'erreur
```
e = y_désiré - y_actuel
```

**Étape 2 :** Calculer les variations
```
Δw₁ = η · e · x₁
Δw₂ = η · e · x₂
Δw₀ = η · e · 1
```

**Étape 3 :** Mettre à jour
```
w₁_nouveau = w₁_ancien + Δw₁
w₂_nouveau = w₂_ancien + Δw₂
w₀_nouveau = w₀_ancien + Δw₀
```

**Étape 4 :** TOUJOURS vérifier
```
Tester avec la même entrée :
S_nouveau = w_nouveau · x + w₀_nouveau
y_nouveau = f(S_nouveau)
Comparer avec y_désiré
```

**INTERPRÉTATION DES SIGNES :**
- `e > 0` : sortie trop faible → augmenter les poids
- `e < 0` : sortie trop haute → diminuer les poids
- `e = 0` : correct → pas de mise à jour

**PIÈGES À ÉVITER :**
- Oublier de mettre à jour w₀ (le biais!)
- Mauvais signe de l'erreur : e = désiré - actuel (pas l'inverse!)
- Ne pas multiplier par xᵢ
- Ne pas vérifier la nouvelle prédiction

---

### Type 4 : Descente de gradient

**MÉTHODE :**

**Étape 1 :** Calculer le gradient
```
Pour C(w) = (w-a)² :
dC/dw = 2(w-a)

Pour C(w₁,w₂) = w₁² + w₂² :
∂C/∂w₁ = 2w₁
∂C/∂w₂ = 2w₂
```

**Étape 2 :** Mettre à jour (ATTENTION AU SIGNE !) :
```
w_nouveau = w_ancien - η · gradient

Exemple 1D :
Si w=0, dC/dw=-6, η=0.1
w_nouveau = 0 - 0.1·(-6) = 0.6

Exemple 2D :
Si (w₁,w₂)=(4,2), ∇C=(8,8), η=0.1
w₁_nouveau = 4 - 0.1·8 = 3.2
w₂_nouveau = 2 - 0.1·8 = 1.2
```

**Étape 3 :** Calculer C(w_nouveau) pour vérifier
```
C devrait DIMINUER à chaque itération
Si C augmente → η trop grand!
```

**PIÈGES À ÉVITER :**
- Mauvais signe : c'est w - η·gradient (MOINS!)
- Oublier la règle de chaîne pour les dérivées
- Ne pas vérifier que C diminue
- η trop grand → oscillation/divergence

---

## ERREURS FRÉQUENTES (À NE JAMAIS FAIRE)

### Erreur #1 : Oublier le biais
```
FAUX : S = w₁x₁ + w₂x₂
JUSTE : S = w₁x₁ + w₂x₂ + w₀
```

### Erreur #2 : Mauvais signe de mise à jour
```
FAUX : w_nouveau = w_ancien + η · gradient (descente!)
JUSTE : w_nouveau = w_ancien - η · gradient
```

### Erreur #3 : Confondre Hebb et Delta
```
FAUX : Règle de Hebb avec erreur
JUSTE : 
   Delta : Δw = η · e · x (avec erreur)
   Hebb  : Δw = η · y · x (sans erreur)
```

### Erreur #4 : Erreur dans le mauvais sens
```
FAUX : e = y_actuel - y_désiré
JUSTE : e = y_désiré - y_actuel
```

### Erreur #5 : Oublier x₀ = 1 pour le biais
```
FAUX : Δw₀ = η · e
JUSTE : Δw₀ = η · e · 1 (x₀ = 1 toujours)
```

---

## TABLE DE DÉCISION RAPIDE

### Quelle fonction d'activation utiliser ?

| Situation | Fonction | Pourquoi |
|-----------|----------|----------|
| Sortie binaire (0 ou 1) | Heaviside ou Sigmoid | Classification binaire |
| Couche cachée (réseau profond) | ReLU | Pas de gradient évanescent |
| Sortie doit être centrée sur 0 | Tanh | Range (-1, 1) |
| Besoin de dérivée partout | Sigmoid ou Tanh | Différentiable partout |
| Vitesse de calcul critique | ReLU | Très rapide (max operation) |

---

### Quel taux d'apprentissage η choisir ?

| Comportement observé | Problème | Solution |
|---------------------|----------|----------|
| Poids changent à peine | η trop petit | Augmenter η |
| Coût oscille sans converger | η trop grand | Diminuer η |
| Coût AUGMENTE | η BEAUCOUP trop grand | Diviser η par 10 |
| Convergence lente mais stable | η correct mais petit | Acceptable ou augmenter légèrement |
| Convergence rapide et stable | η optimal | Garder cette valeur! |

**Valeurs typiques :** η ∈ [0.01, 0.5] pour perceptrons

---

## 🔢 VALEURS À MÉMORISER

**Constantes mathématiques :**
```
e ≈ 2.718
√2 ≈ 1.414
```

**Sigmoïde - valeurs clés :**
```
σ(0) = 0.5
σ(1) ≈ 0.73
σ(2) ≈ 0.88
σ(-1) ≈ 0.27
σ(-2) ≈ 0.12
```

**Dérivées - valeurs clés :**
```
σ'(0) = 0.25 (maximum)
σ'(2) ≈ 0.10
σ'(5) ≈ 0.007 (gradient évanescent!)
```

---

**Formules mémorisées :**
- [ ] Équation perceptron : S = Σwᵢxᵢ + w₀
- [ ] Règle Delta : Δw = η·e·x
- [ ] Descente gradient : w_new = w_old - η·∇C
- [ ] σ'(x) = σ(x)(1-σ(x))
- [ ] tanh'(x) = 1 - tanh²(x)

**Compétences techniques :**
- [ ] Je peux calculer la sortie d'un perceptron en < 2 min
- [ ] Je connais σ(0), σ'(0), tanh(0) par cœur
- [ ] Je peux faire une mise à jour Delta sans regarder la formule
- [ ] Je sais quand gradient descent diverge (η trop grand)
- [ ] Je peux expliquer pourquoi XOR n'est pas linéairement séparable

---

**L'examen teste 3 choses :**

1. **Connaissance des formules** (30% des points)
   → Antisèche + répétition

2. **Capacité de calcul** (50% des points)
   → Practice problems + vérification

3. **Compréhension conceptuelle** (20% des points)
   → Pourquoi les choses fonctionnent

**Tu as les outils. Maintenant TRAVAILLE.**

**BONNE CHANCE! 🚀**