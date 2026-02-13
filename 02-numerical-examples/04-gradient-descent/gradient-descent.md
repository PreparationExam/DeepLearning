# Problèmes de Gradient Descent

## Problème 4.1 : Bases du Gradient Descent en 1D 

**Données :**
- Fonction de coût : C(w) = (w - 3)
- Poids initial : w = 0
- Taux d'apprentissage : η = 0.4

**Objectif :** Effectuer 5 itérations de gradient descent. Montrer la convergence.

---

### Solution :

**Étape 1 : Calculer le gradient**

```
C(w) = (w - 3)

dC/dw = 2(w - 3) · 1 = 2(w - 3)
```

**Étape 2 : Règle de mise à jour du gradient descent**

```
w_new = w_old - η · (dC/dw)
```

---

**ITÉRATION 1 :**

```
w = 0
C(0) = (0-3) = 9

dC/dw|_{w=0} = 2(0 - 3) = -6

w = 0 - 0.4·(-6) = 0 + 2.4 = 2.4
```

---

**ITÉRATION 2 :**

```
w = 2.4
C(2.4) = (2.4-3) = 0.36

dC/dw|_{w=2.4} = 2(2.4 - 3) = -1.2

w = 2.4 - 0.4·(-1.2) = 2.4 + 0.48 = 2.88
```

---

**ITÉRATION 3 :**

```
w = 2.88
C(2.88) = (2.88-3) = 0.0144

dC/dw|_{w=2.88} = 2(2.88 - 3) = -0.24

w = 2.88 - 0.4·(-0.24) = 2.88 + 0.096 = 2.976
```

---

**ITÉRATION 4 :**

```
w = 2.976
C(2.976) = (2.976-3) = 0.000576

dC/dw|_{w=2.976} = 2(2.976 - 3) = -0.048

w = 2.976 - 0.4·(-0.048) = 2.976 + 0.0192 = 2.9952
```

---

**ITÉRATION 5 :**

```
w = 2.9952
C(2.9952) = (2.9952-3)  0.000023

dC/dw|_{w=2.9952} = 2(2.9952 - 3)  -0.0096

w = 2.9952 - 0.4·(-0.0096)  2.99904
```

---

**Tableau récapitulatif :**

| Itération | w | C(w) | dC/dw |
|-----------|---------|----------|--------|
| 0 | 0 | 9 | -6 |
| 1 | 2.4 | 0.36 | -1.2 |
| 2 | 2.88 | 0.0144 | -0.24 |
| 3 | 2.976 | 0.000576 | -0.048 |
| 4 | 2.9952 | 0.000023 | -0.0096 |
| 5 | 2.99904 | ~0 | ~0 |

**Convergence vers w* = 3 (le minimum)**

---

### Analyse :

**Observations :**
1. **Le coût décroît de manière monotone :** 9  0.36  0.0144  ...  0
2. **La magnitude du gradient décroît :** -6  -1.2  -0.24  ...  0
3. **Convergence vers w* = 3** (où dC/dw = 0)
4. **Convergence exponentielle** pour les fonctions quadratiques

**Pourquoi ça fonctionne :**
- Le gradient est NÉGATIF quand w < 3  la mise à jour déplace vers la DROITE
- Le gradient est POSITIF quand w > 3  la mise à jour déplace vers la GAUCHE
- Le gradient est NUL à w = 3  plus de mises à jour (convergé !)

---

### Erreurs courantes :

**Mauvaise direction de mise à jour** - Ce doit être w_new = w - η·gradient (signe MOINS !)
**Oubli de la règle de chaîne** - dC/dw pour (w-3) nécessite la règle de chaîne
**Ne pas montrer que le coût décroît** - Cela prouve la convergence
**S'arrêter trop tôt** - Continuer jusqu'à ce que le gradient  0

---

**Temps de résolution :** 12-15 minutes
**Fréquence à l'examen :** 90%

---

## Problème 4.2 : Impact du taux d'apprentissage

**Même fonction :** C(w) = (w - 3)
**Même initialisation :** w = 0

**Comparer 3 itérations avec :**
a) η = 0.1 (petit)
b) η = 0.4 (modéré)
c) η = 1.1 (trop grand)

---

### Solution :

**Gradient :** dC/dw = 2(w - 3)

---

**a) η = 0.1 (Petit taux d'apprentissage) :**

```
Iteration 1:
dC/dw = 2(0-3) = -6
w = 0 - 0.1·(-6) = 0.6

Iteration 2:
dC/dw = 2(0.6-3) = -4.8
w = 0.6 - 0.1·(-4.8) = 1.08

Iteration 3:
dC/dw = 2(1.08-3) = -3.84
w = 1.08 - 0.1·(-3.84) = 1.464
```

**Résultat :** Progression LENTE mais régulière (0  0.6  1.08  1.464)

---

**b) η = 0.4 (Modéré - du Problème 4.1) :**

```
w = 2.4
w = 2.88
w = 2.976
```

**Résultat :** Convergence RAPIDE, presque au minimum

---

**c) η = 1.1 (Trop grand) :**

```
Iteration 1:
dC/dw = 2(0-3) = -6
w = 0 - 1.1·(-6) = 6.6

Iteration 2:
dC/dw = 2(6.6-3) = 7.2
w = 6.6 - 1.1·(7.2) = 6.6 - 7.92 = -1.32

Iteration 3:
dC/dw = 2(-1.32-3) = -8.64
w = -1.32 - 1.1·(-8.64) = -1.32 + 9.504 = 8.184
```

**Résultat :** OSCILLATION ! (0  6.6  -1.32  8.184  ...)
**DIVERGENCE - ça EMPIRE !**

---

### Comparaison des coûts :

| Itération | η=0.1 | η=0.4 | η=1.1 |
|-----------|--------|--------|--------|
| C(w) | 9 | 9 | 9 |
| C(w) | 5.76 | 0.36 | 12.96  |
| C(w) | 3.686 | 0.0144 | 18.66  |
| C(w) | 2.359 | 0.000576 | 26.88  |

**Conclusion :**
- η = 0.1 : Convergence lente, mais stable
- η = 0.4 : Convergence rapide, optimal
- η = 1.1 : DIVERGE (coût croissant !)

---

### Interprétation visuelle :

```
C(w)
  |     
9 |                           (η=1.1 rebondit)
  |  \                /
  |    \            /
  |      \        /
  |              /         
  |         \   /
0 |________________________ w
            3 (minimum)

 Petits pas (η=0.1)
 Pas moyens (η=0.4)  
 Dépassement (η=1.1)
```

---

### Erreurs courantes :

**Penser que plus grand est toujours mieux** - Un grand η provoque la divergence
**Ne pas vérifier si le coût décroît** - C'est la validation
**Ignorer l'oscillation** - Signe que η est trop grand

---

**Seuil critique :** Pour C(w) = (w-a), le gradient descent diverge quand η > 1.0

**Principe général :** 
- η trop petit : Lent mais sûr
- η optimal : Convergence rapide
- η trop grand : Oscillation et divergence

**Temps de résolution :** 15-18 minutes
**Fréquence à l'examen :** 70%

---

## Problème 4.3 : Gradient Descent en 2D

**Données :**
- Fonction de coût : C(w, w) = w + 2w
- Point initial : (w, w) = (4, 2)
- Taux d'apprentissage : η = 0.1

**Objectif :** Effectuer 2 itérations de gradient descent.

---

### Solution :

**Étape 1 : Calculer le gradient**

Le gradient est le vecteur des dérivées partielles :

```
C = (C/w, C/w)

C/w = 2w
C/w = 4w

C = (2w, 4w)
```

**Étape 2 : Règle de mise à jour**

```
w_new = w_old - η · C
```

Par composantes :
```
w_new = w_old - η · (C/w)
w_new = w_old - η · (C/w)
```

---

**ITÉRATION 1 :**

```
Actuel : (w, w) = (4, 2)
Coût : C(4, 2) = 4 + 2·2 = 16 + 8 = 24

Gradient :
C/w = 2·4 = 8
C/w = 4·2 = 8
C = (8, 8)

Mise à jour :
w_new = 4 - 0.1·8 = 4 - 0.8 = 3.2
w_new = 2 - 0.1·8 = 2 - 0.8 = 1.2

Nouveau point : (3.2, 1.2)
```

---

**ITÉRATION 2 :**

```
Actuel : (w, w) = (3.2, 1.2)
Coût : C(3.2, 1.2) = 3.2 + 2·1.2 = 10.24 + 2.88 = 13.12

Gradient :
C/w = 2·3.2 = 6.4
C/w = 4·1.2 = 4.8
C = (6.4, 4.8)

Mise à jour :
w_new = 3.2 - 0.1·6.4 = 3.2 - 0.64 = 2.56
w_new = 1.2 - 0.1·4.8 = 1.2 - 0.48 = 0.72

Nouveau point : (2.56, 0.72)
```

---

**Résumé :**

| Itération | (w, w) | C(w, w) | C |
|-----------|----------|-----------|---------|
| 0 | (4, 2) | 24 | (8, 8) |
| 1 | (3.2, 1.2) | 13.12 | (6.4, 4.8) |
| 2 | (2.56, 0.72) | 7.59 | (5.12, 2.88) |

**Convergence vers (0, 0), où C est minimum (C = 0)**

---

### Interprétation géométrique :

**La fonction C(w, w) = w + 2w est un paraboloïde elliptique.**

Courbes de niveau (C constant) :
```
     w
      |
    2 |              Courbes de niveau :
      |   /           C = 24 (ellipse extérieure)
    1 |              C = 13 (milieu)
      | /             C = 8 (intérieure)
    0 |____________ w
      0   2   4
      
Le point de départ (4,2) se déplace vers l'origine (0,0)
```

**Le gradient C pointe dans la direction de la PLUS FORTE ASCENSION.**
**On se déplace dans la direction OPPOSÉE (plus forte descente).**

---

### Erreurs courantes :

**Oublier que c'est une mise à jour vectorielle** - Il faut mettre à jour LES DEUX composantes
**Ne pas calculer correctement les dérivées partielles** - (2w)/w = 4w, pas 2w
**Ajouter au lieu de soustraire le gradient** - C'est une descente, pas une montée !

---

**Point clé :** 
Le gradient descent en 2D (ou plus) consiste simplement à appliquer la règle 1D à chaque dimension indépendamment.

**Temps de résolution :** 12-15 minutes

**Fréquence à l'examen :** 60%

---

## Problème 4.4 : Convexe vs Non-Convexe

**Comparer deux fonctions :**

**Fonction A (Convexe) :** C(w) = w
**Fonction B (Non-convexe) :** C(w) = w - 4w

**Objectif :** 
1. Trouver tous les points critiques (où dC/dw = 0)
2. Classifier comme minimum, maximum ou point selle
3. Exécuter le gradient descent depuis w = -1 pour les deux fonctions (η = 0.1, 3 itérations)

---

### Solution :

---

**FONCTION A : C(w) = w**

**Étape 1 : Points critiques**

```
dC/dw = 2w = 0
w* = 0
```

**Test de la dérivée seconde :**
```
dC/dw = 2 > 0  MINIMUM
```

**Un seul point critique : w* = 0 (minimum global)**

---

**FONCTION B : C(w) = w - 4w**

**Étape 1 : Points critiques**

```
dC/dw = 4w - 8w = 4w(w - 2) = 0

Solutions :
w = 0  OU  w = 2
w = 0  OU  w = 2

Trois points critiques : w*  {-2, 0, +2}
```

**Étape 2 : Test de la dérivée seconde**

```
dC/dw = 12w - 8

En w = 0 :
dC/dw = -8 < 0  MAXIMUM LOCAL

En w = 2 :
dC/dw = 12·2 - 8 = 16 > 0  MINIMUM LOCAL
C(2) = (2) - 4(2) = 4 - 8 = -4

En w = -2 :
dC/dw = 16 > 0  MINIMUM LOCAL
C(-2) = -4
```

**Deux minima locaux en w = 2 (les deux avec C = -4)**
**Un maximum local en w = 0 (avec C = 0)**

---

**Étape 3 : Gradient descent depuis w = -1**

**Fonction A : C(w) = w**

```
Iteration 1:
dC/dw = 2(-1) = -2
w = -1 - 0.1·(-2) = -0.8

Iteration 2:
dC/dw = 2(-0.8) = -1.6
w = -0.8 - 0.1·(-1.6) = -0.64

Iteration 3:
dC/dw = 2(-0.64) = -1.28
w = -0.64 - 0.1·(-1.28) = -0.512
```

**Convergence vers w* = 0 (le SEUL minimum)**

---

**Fonction B : C(w) = w - 4w**

```
dC/dw = 4w - 8w

Iteration 1:
w = -1
dC/dw = 4(-1) - 8(-1) = -4 + 8 = 4
w = -1 - 0.1·4 = -1.4

Iteration 2:
dC/dw = 4(-1.4) - 8(-1.4) = -10.976 + 11.2 = 0.224
w = -1.4 - 0.1·0.224 = -1.4224

Iteration 3:
dC/dw = 4(-1.4224) - 8(-1.4224)  -11.48 + 11.38 = -0.1
w  -1.4224 - 0.1·(-0.1)  -1.412
```

**Convergence vers w* = -2  -1.414 (minimum local à GAUCHE)**

---

### Différence clé :

**CONVEXE (Fonction A) :**
- UN SEUL minimum global
- Le gradient descent le trouve TOUJOURS
- Peu importe le point de départ
- Convergence garantie

**NON-CONVEXE (Fonction B) :**
- PLUSIEURS minima locaux
- Le gradient descent trouve le minimum local le PLUS PROCHE
- Le point de départ COMPTE
- Si on commence à w = +1, on convergerait vers w* = +2 à la place !
- Pas de garantie de trouver le minimum GLOBAL

---

### Comparaison visuelle :

```
Fonction A (Convexe) :
C(w)
  |
  |    \      /
  |     \    /
  |      \  /
  |       \/
  |________________ w
          0

Fonction B (Non-convexe) :
C(w)
  |      
  |    /\    /\
  | /\/  \/  \/  \
  |/          \
  |_______________ w
   -2   0   +2
   min  max  min
```

---

### Erreurs courantes :

**Penser que le gradient descent trouve toujours le minimum global** - Seulement pour les fonctions convexes !
**Ne pas trouver tous les points critiques** - Il faut résoudre dC/dw = 0 complètement
**Confondre minima locaux et globaux** - Les fonctions non-convexes ont les deux
**Ne pas utiliser le test de la dérivée seconde** - Nécessaire pour classifier les points critiques

---

**Question en or pour l'examen :**

**Q : "Pourquoi la convexité est-elle importante en optimisation ?"**

**R :** 
1. Les fonctions convexes ont un minimum global UNIQUE
2. Le gradient descent est GARANTI de converger
3. Le point de départ n'a pas d'importance
4. Pas de risque de rester bloqué dans des minima locaux

**Q : "Que se passe-t-il avec les fonctions non-convexes ?"**

**R :**
1. Il existe plusieurs minima locaux
2. Le gradient descent trouve le minimum local le PLUS PROCHE
3. Il peut ne pas trouver le minimum GLOBAL
4. Stratégies nécessaires : redémarrages aléatoires, momentum, mises à jour stochastiques

**Temps de résolution :** 25-30 minutes
**Difficulté :**
**Fréquence à l'examen :** 50% (mais compréhension conceptuelle CRITIQUE)

---

## Problème 4.5 : Perceptron comme Gradient Descent 

**Montrer que la Delta rule est du gradient descent sur la Mean Squared Error (MSE).**

**Données :**
- Sortie du perceptron : ŷ = σ(w·x + w) où σ est la sigmoid
- Coût MSE : C = (1/2)(y - ŷ) pour un seul exemple
- Prouver : Δw = η · C/w = η · (y - ŷ) · σ'(z) · x

---

### Solution :

**Étape 1 : Définir la notation**

```
z = w·x + w (somme pondérée)
ŷ = σ(z) (sortie prédite)
y = sortie désirée
C = (1/2)(y - ŷ) (coût pour un exemple)
```

**Étape 2 : Calculer C/w en utilisant la règle de chaîne**

```
C/w = C/ŷ · ŷ/z · z/w
```

**Étape 3 : Calculer chaque terme**

**Terme 1 : C/ŷ**

```
C = (1/2)(y - ŷ)

C/ŷ = (1/2) · 2(y - ŷ) · (-1) = -(y - ŷ)
```

**Terme 2 : ŷ/z**

```
ŷ = σ(z)

ŷ/z = σ'(z) = σ(z)(1 - σ(z))
```

**Terme 3 : z/w**

```
z = w·x + w = Σwᵢxᵢ + w

z/wᵢ = xᵢ
```

**Étape 4 : Combiner avec la règle de chaîne**

```
C/wᵢ = C/ŷ · ŷ/z · z/wᵢ
       = -(y - ŷ) · σ'(z) · xᵢ
       = -(y - ŷ) · σ(z)(1 - σ(z)) · xᵢ
```

**Étape 5 : Mise à jour par gradient descent**

```
Δwᵢ = -η · C/wᵢ
    = -η · [-(y - ŷ) · σ'(z) · xᵢ]
    = η · (y - ŷ) · σ'(z) · xᵢ
```

**Étape 6 : Cas particulier - Activation linéaire**

Si on utilise une **activation linéaire** (ou Heaviside où σ' = 1) :

```
Δwᵢ = η · (y - ŷ) · xᵢ
    = η · e · xᵢ
```

**C'est exactement la Delta rule !**

---

### Points clés :

1. **La Delta rule est du gradient descent** sur la fonction de coût MSE
2. **Le terme d'erreur (y - ŷ)** provient de C/ŷ
3. **L'entrée xᵢ** provient de z/wᵢ
4. **La dérivée de l'activation σ'(z)** provient de ŷ/z
5. **La backpropagation moderne** est simplement cette règle de chaîne étendue à plusieurs couches

---

### Lien avec le Gradient Descent général :

**Forme générale :**
```
w_new = w_old - η · C(w)
```

**Pour le perceptron :**
```
wᵢ_new = wᵢ_old - η · C/wᵢ
       = wᵢ_old + η · (y - ŷ) · σ'(z) · xᵢ
```

**Cela unifie :**
- Gradient descent (optimisation générale)
- Delta rule (apprentissage du perceptron)
- Backpropagation (deep learning)

Tout repose sur la même idée fondamentale : **ajuster les paramètres proportionnellement à leur impact sur l'erreur.**

---

### Erreurs courantes :

**Ne pas utiliser la règle de chaîne** - Il faut décomposer C/w en composantes
**Oublier le signe négatif** - C/ŷ = -(y - ŷ), pas +(y - ŷ)
**Confondre la mise à jour du gradient descent** - w_new = w_old - ηC (MOINS le gradient)
**Ne pas voir le lien** - La Delta rule EST du gradient descent !

---

**Pourquoi c'est important :**

Ce problème montre que :
1. L'apprentissage du perceptron est une optimisation rigoureuse
2. Tout ce que nous avons appris (gradient, taux d'apprentissage, convergence) s'applique
3. C'est la base pour comprendre la backpropagation au Chapitre 2
4. Le deep learning = gradient descent sur des fonctions de coût complexes

**Temps de résolution :** 20-25 minutes
**Difficulté :** 
**Fréquence à l'examen :** 40% (mais teste une compréhension profonde)

---

## Résumé : Maîtrise du Gradient Descent

**Formule fondamentale :**
```
w_new = w_old - η · C(w)
```

**Concepts clés :**
1. **Le gradient C** pointe dans la direction de la plus forte augmentation
2. **Le gradient négatif** pointe vers la plus forte diminution
3. **Le taux d'apprentissage η** contrôle la taille du pas
4. **Convergence** quand le gradient  0 (point critique)

**Types de problèmes :**
1. **Optimisation 1D** - Calculer dC/dw, itérer les mises à jour
2. **Choix du taux d'apprentissage** - Trop petit = lent, trop grand = diverge
3. **2D/Multi-dimensionnel** - Calculer les dérivées partielles, mises à jour vectorielles
4. **Convexe vs non-convexe** - Comprendre minima locaux vs globaux
5. **Lien avec les règles d'apprentissage** - La Delta rule est du gradient descent

**Liste de vérification pour l'examen :**
-  Savoir calculer le gradient de fonctions quadratiques
-  Savoir effectuer des mises à jour itératives
-  Savoir identifier quand le taux d'apprentissage est trop grand (oscillation/divergence)
-  Comprendre que convexe = un minimum, non-convexe = plusieurs minima
-  Savoir dériver la Delta rule à partir du gradient descent

**VOUS MAÎTRISEZ MAINTENANT COMPLÈTEMENT LES EXERCICES NUMÉRIQUES DU CHAPITRE 1.**

