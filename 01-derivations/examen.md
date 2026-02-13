# Questions d'Examen - Dérivations Mathématiques
## Chapitre 1 : Réseaux de Neurones

**Instructions pour vos coéquipiers :**
Ces questions nécessitent des dérivations complètes et une compréhension mathématique approfondie. Chaque question vaut des points importants en examen. Pratiquez-les jusqu'à maîtrise complète !

---

## ⭐ Question 1 : Dérivée de la Fonction Sigmoïde (15 points)

**Énoncé :**
Soit la fonction sigmoïde définie par :
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**a)** Démontrez rigoureusement que la dérivée de la fonction sigmoïde est :
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**b)** Utilisez ce résultat pour calculer $\sigma'(0)$.

**c)** Montrez que $\sigma'(x) \leq \frac{1}{4}$ pour tout $x \in \mathbb{R}$. Que signifie cette propriété pour l'apprentissage par gradient ?

---

## ⭐ Question 2 : Dérivée de la Fonction Tanh (12 points)

**Énoncé :**
Soit la fonction tangente hyperbolique :
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**a)** Démontrez que :
$$\tanh'(x) = 1 - \tanh^2(x)$$

**b)** Montrez la relation suivante :
$$\tanh(x) = 2\sigma(2x) - 1$$

où $\sigma$ est la fonction sigmoïde.

**c)** Déduisez de cette relation une expression alternative pour $\tanh'(x)$ en fonction de $\sigma$.

---

## ⭐ Question 3 : Règle Delta - Dérivation Complète (20 points)

**Énoncé :**
Considérez un perceptron avec :
- Entrées : $x = [x_1, x_2, \ldots, x_n]^T$
- Poids : $w = [w_1, w_2, \ldots, w_n]^T$
- Biais : $w_0$
- Fonction d'activation : $f$
- Fonction de coût : $E = \frac{1}{2}(y^d - \hat{y})^2$

où la sortie prédite est :
$$\hat{y} = f\left(\sum_{i=1}^{n} x_i w_i + w_0\right)$$

**a)** En utilisant la règle de dérivation en chaîne, calculez $\frac{\partial E}{\partial w_j}$ pour $j = 1, 2, \ldots, n$.

**b)** Dérivez la règle de mise à jour des poids :
$$w_j^{\text{nouveau}} = w_j^{\text{ancien}} + \eta \delta x_j$$

et donnez l'expression explicite de $\delta$.

**c)** Cas particulier : Si $f(z) = z$ (activation linéaire), que devient la règle delta ?

**d)** Dérivez également la règle de mise à jour pour le biais $w_0$.

---

## ⭐ Question 4 : Gradient de la MSE (18 points)

**Énoncé :**
Soit un modèle linéaire $\hat{y}_i = w^T x_i$ et la fonction de coût MSE :
$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (y_i^d - w^T x_i)^2$$

**a)** Calculez le gradient $\nabla_w J(w)$ en développant complètement le calcul.

**b)** Exprimez le résultat sous forme matricielle en utilisant :
$$X = \begin{pmatrix}
— x_1^T — \\
— x_2^T — \\
\vdots \\
— x_m^T —
\end{pmatrix}, \quad Y = \begin{pmatrix}
y_1^d \\
y_2^d \\
\vdots \\
y_m^d
\end{pmatrix}$$

**c)** Donnez la règle de mise à jour des poids pour la descente de gradient en utilisant cette forme matricielle.

---

## ⭐ Question 5 : Analyse de Convexité (16 points)

**Énoncé :**
Considérez la fonction de coût quadratique pour un modèle linéaire :
$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - w^T x_i)^2$$

**a)** Calculez la matrice Hessienne $H$ de $J(w)$ :
$$H_{jk} = \frac{\partial^2 J}{\partial w_j \partial w_k}$$

**b)** Exprimez $H$ sous forme matricielle en fonction de la matrice de données $X$.

**c)** Démontrez que $H$ est semi-définie positive, et concluez sur la convexité de $J(w)$.

**d)** Quelle est l'implication pratique de cette convexité pour l'apprentissage du modèle ?

---

## ⭐ Question 6 : Perceptron et Hyperplan (14 points)

**Énoncé :**
Un perceptron binaire avec deux entrées est défini par :
$$\hat{y} = \text{signe}(w_1 x_1 + w_2 x_2 + w_0)$$

**a)** Déterminez l'équation de l'hyperplan de décision (frontière entre les deux classes).

**b)** Pour $w_1 = 2$, $w_2 = -1$, $w_0 = 3$, tracez cet hyperplan dans le plan $(x_1, x_2)$.

**c)** Testez les points suivants et déterminez leur classe :
- $A = (0, 0)$
- $B = (2, 1)$
- $C = (-1, 4)$
- $D = (1, 5)$

**d)** Calculez la distance du point $P = (1, 1)$ à l'hyperplan de décision. 

**Rappel :** Distance d'un point $(x_0, y_0)$ à la droite $ax + by + c = 0$ :
$$d = \frac{|ax_0 + by_0 + c|}{\sqrt{a^2 + b^2}}$$

---

## ⭐ Question 7 : Propagation du Gradient avec ReLU (13 points)

**Énoncé :**
Considérez un neurone avec fonction d'activation ReLU :
$$\text{ReLU}(x) = \max(0, x)$$

et fonction de coût :
$$E = \frac{1}{2}(y^d - \hat{y})^2$$

où :
$$\hat{y} = \text{ReLU}\left(\sum_{i=1}^{n} w_i x_i + w_0\right)$$

**a)** Rappelez la dérivée de la fonction ReLU :
$$\frac{d(\text{ReLU})}{dx}(x) = ?$$

**b)** Calculez $\frac{\partial E}{\partial w_j}$ en utilisant la règle de la chaîne.

**c)** Que se passe-t-il si $\sum_{i=1}^{n} w_i x_i + w_0 < 0$ ? Comment cela affecte-t-il l'apprentissage ?

**d)** Ce phénomène est appelé "dying ReLU". Expliquez pourquoi et proposez une solution.

---

## ⭐ Question 8 : Comparaison des Méthodes de Descente (15 points)

**Énoncé :**
Soit un ensemble d'entraînement de $m = 1000$ exemples.

**a)** Comparez les trois variantes de descente de gradient :

| Méthode | Formule du gradient | Nombre de calculs par itération |
|---------|---------------------|--------------------------------|
| Batch GD | ? | ? |
| Stochastic GD | ? | ? |
| Mini-Batch GD (batch = 50) | ? | ? |

**b)** Pour chaque méthode, calculez le nombre total de calculs de gradient nécessaires pour effectuer une époque complète (passage sur tous les exemples).

**c)** Discutez des avantages et inconvénients de chaque approche en termes de :
- Vitesse de convergence
- Stabilité
- Coût de calcul

---

## ⭐ Question 9 : Loi de Hebb et Apprentissage (12 points)

**Énoncé :**
Selon la loi de Hebb, la modification du poids synaptique entre deux neurones $i$ et $j$ est :
$$\Delta w_{ij} = \eta \cdot a_i \cdot a_j$$

où $a_i$ et $a_j$ sont les activations des neurones.

**a)** Expliquez en quoi cette règle diffère de la règle delta.

**b)** Considérez quatre cas d'activation :
1. $a_i = 1, a_j = 1$
2. $a_i = 1, a_j = 0$
3. $a_i = 0, a_j = 1$
4. $a_i = 0, a_j = 0$

Pour chaque cas, calculez $\Delta w_{ij}$ avec $\eta = 0.1$.

**c)** La loi de Hebb est-elle **supervisée** ou **non supervisée** ? Justifiez.

**d)** Quel est le problème majeur de la loi de Hebb telle que formulée ci-dessus ? (Indice : pensez à la croissance des poids)

---

## ⭐ Question 10 : Problème Complet - XOR (25 points)

**Énoncé :**
Le problème XOR (OU exclusif) est défini par :

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**a)** Montrez qu'un perceptron simple **ne peut pas** résoudre le problème XOR en prouvant qu'il n'existe pas de droite séparant correctement les classes dans le plan $(x_1, x_2)$.

**b)** Proposez une architecture de réseau à **deux couches** capable de résoudre XOR :
- Couche cachée : 2 neurones avec activation sigmoïde
- Couche de sortie : 1 neurone avec activation sigmoïde

Donnez un jeu de poids $w_{ij}$ permettant de résoudre le problème.

**c)** Pour votre architecture, calculez la sortie $\hat{y}$ pour chacun des quatre exemples d'entraînement.

**d)** Si on utilise la fonction de coût MSE :
$$J = \frac{1}{4} \sum_{i=1}^{4} (y_i^d - \hat{y}_i)^2$$

Calculez $J$ pour votre réseau. Est-ce satisfaisant ?

**e)** Expliquez pourquoi ce problème illustre les limites du perceptron simple et l'importance des réseaux multicouches.

---

## 📝 Barème et Conseils

### Répartition des Points
- **Total : 160 points**
- Temps estimé : 3 heures
- Note finale : $\frac{\text{Points obtenus}}{160} \times 20$

### Critères d'Évaluation
1. **Rigueur mathématique** (40%) : Toutes les étapes doivent être justifiées
2. **Clarté de la présentation** (20%) : Notation correcte, calculs organisés
3. **Résultats numériques** (20%) : Calculs exacts, pas d'erreurs d'arithmétique
4. **Interprétation** (20%) : Comprendre la signification des résultats

### Conseils pour Réussir
✅ Maîtrisez parfaitement les dérivées des fonctions d'activation
✅ Pratiquez la règle de la chaîne jusqu'à l'automatisme
✅ Comprenez géométriquement les hyperplans et la convexité
✅ Entraînez-vous à passer du scalaire au matriciel
✅ Relisez vos dérivations pour éviter les erreurs de signe

---

## 🎯 Questions Bonus (Points Supplémentaires)

### Bonus 1 (5 points)
Démontrez que la fonction sigmoïde peut être exprimée en fonction de tanh :
$$\sigma(x) = \frac{1 + \tanh(x/2)}{2}$$

### Bonus 2 (5 points)
Calculez la dérivée seconde de la fonction sigmoïde : $\sigma''(x)$, et déterminez ses points d'inflexion.

### Bonus 3 (10 points)
Proposez une fonction d'activation qui ne souffre **ni** du problème de disparition du gradient **ni** du problème des neurones morts. Justifiez mathématiquement vos affirmations.

---

**Bonne chance ! 🚀**

**Note pour vos coéquipiers :**
- Travaillez ces questions en groupe
- Vérifiez vos réponses mutuellement
- Créez des fiches récapitulatives des dérivations clés
- Simulez des conditions d'examen (temps limité, sans notes)