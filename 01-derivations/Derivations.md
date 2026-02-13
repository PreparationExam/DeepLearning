# Dérivations Mathématiques - Réseaux de Neurones
## Chapitre 1 : Concepts Fondamentaux

---

## Table des matières
1. [Neurone Formel de McCulloch et Pitts](#1-neurone-formel-de-mcculloch-et-pitts)
2. [Fonctions d'Activation](#2-fonctions-dactivation)
3. [Le Perceptron](#3-le-perceptron)
4. [Fonction de Coût](#4-fonction-de-coût)
5. [Gradient d'une Fonction](#5-gradient-dune-fonction)
6. [Dérivation de la Règle Delta](#6-dérivation-de-la-règle-delta)
7. [Algorithme de Descente de Gradient](#7-algorithme-de-descente-de-gradient)
8. [Propriétés de Convexité](#8-propriétés-de-convexité)

---

## 1. Neurone Formel de McCulloch et Pitts

### 1.1 Définition Mathématique

Le neurone formel reçoit un vecteur d'entrées binaires :
$$X = [x_1, x_2, \ldots, x_n]^T$$

où chaque $x_i \in \{0, 1\}$.

### 1.2 Fonction de Sortie

La sortie $y$ est également binaire : $y \in \{0, 1\}$.

### 1.3 Phase d'Agrégation

On calcule la **somme pondérée** des entrées :
$$S = \sum_{i=1}^{n} x_i w_i$$

où $w_i$ représente le **poids synaptique** associé à l'entrée $x_i$.

### 1.4 Phase d'Activation

La sortie est déterminée par une **fonction seuil** (Heaviside) :

$$y = f(S) = H(S - \theta) = \begin{cases}
1 & \text{si } \sum_{i=1}^{n} x_i w_i \geq \theta \\
0 & \text{sinon}
\end{cases}$$

où $\theta$ est le **seuil d'activation**.

---

## 2. Fonctions d'Activation

### 2.1 Fonction Marche d'Escalier (Heaviside)

**Définition :**
$$H(x) = \begin{cases}
0 & \text{si } x < 0 \\
1 & \text{si } x \geq 0
\end{cases}$$

**Propriétés :**
- Non continue
- Non dérivable en $x = 0$
- Utilisée dans le neurone formel original

---

### 2.2 Fonction Sigmoïde

**Définition :**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Dérivée :**

Calculons $\frac{d\sigma}{dx}$ :

$$\frac{d\sigma}{dx} = \frac{d}{dx}\left(\frac{1}{1 + e^{-x}}\right)$$

En utilisant la règle de dérivation des quotients :
$$\frac{d\sigma}{dx} = \frac{0 \cdot (1 + e^{-x}) - 1 \cdot (-e^{-x})}{(1 + e^{-x})^2}$$

$$= \frac{e^{-x}}{(1 + e^{-x})^2}$$

On peut réécrire cela en fonction de $\sigma(x)$ :

$$\frac{d\sigma}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}}$$

$$= \sigma(x) \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x) \cdot \left(1 - \sigma(x)\right)$$

**Résultat important :**
$$\boxed{\sigma'(x) = \sigma(x)(1 - \sigma(x))}$$

**Propriétés :**
- Continue et dérivable partout
- Sortie dans $(0, 1)$
- Forme en "S"
- Facilite l'optimisation par descente de gradient

---

### 2.3 Fonction Tangente Hyperbolique (tanh)

**Définition :**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Dérivée :**

$$\frac{d(\tanh)}{dx} = 1 - \tanh^2(x)$$

**Démonstration :**

Posons $u = e^x - e^{-x}$ et $v = e^x + e^{-x}$

Alors $\tanh(x) = \frac{u}{v}$

$$\frac{d(\tanh)}{dx} = \frac{u'v - uv'}{v^2}$$

Avec $u' = e^x + e^{-x}$ et $v' = e^x - e^{-x}$ :

$$\frac{d(\tanh)}{dx} = \frac{(e^x + e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})(e^x - e^{-x})}{(e^x + e^{-x})^2}$$

$$= \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}$$

$$= 1 - \frac{(e^x - e^{-x})^2}{(e^x + e^{-x})^2} = 1 - \tanh^2(x)$$

**Résultat :**
$$\boxed{\tanh'(x) = 1 - \tanh^2(x)}$$

**Propriétés :**
- Sortie dans $(-1, 1)$
- Fonction impaire : $\tanh(-x) = -\tanh(x)$
- Centrée autour de 0

---

### 2.4 Fonction ReLU (Rectified Linear Unit)

**Définition :**
$$\text{ReLU}(x) = \max(0, x) = \begin{cases}
0 & \text{si } x < 0 \\
x & \text{si } x \geq 0
\end{cases}$$

**Dérivée :**
$$\text{ReLU}'(x) = \begin{cases}
0 & \text{si } x < 0 \\
1 & \text{si } x > 0 \\
\text{indéfini} & \text{si } x = 0
\end{cases}$$

**Propriétés :**
- Calcul très simple et rapide
- Évite le problème de disparition du gradient
- Non-saturante pour $x > 0$
- **Fonction la plus utilisée en Deep Learning moderne**

---

## 3. Le Perceptron

### 3.1 Modèle Mathématique

Le perceptron calcule :
$$\hat{y}_i = f\left(\sum_{j=1}^{n} x_j w_j + w_0\right)$$

où :
- $x_j$ : les entrées du modèle
- $w_j$ : les poids associés à chaque entrée
- $w_0$ : le biais (ou seuil)
- $f$ : la fonction d'activation

### 3.2 Forme Vectorielle

On peut écrire cela sous forme vectorielle en incluant le biais :

$$\hat{y} = f(w^T x + w_0)$$

ou en ajoutant une entrée constante $x_0 = 1$ :

$$\hat{y} = f\left(\sum_{i=0}^{n} x_i w_i\right) = f(w^T x)$$

---

### 3.3 Perceptron comme Classifieur Linéaire

Le perceptron sépare linéairement l'espace d'entrée en deux classes selon :

$$\text{signe}(x_1, \ldots, x_n) = \begin{cases}
1 & \text{si } \sum_{i=1}^{n} w_i x_i + w_0 > 0 \\
-1 & \text{sinon}
\end{cases}$$

### 3.4 Équation de l'Hyperplan de Décision

La frontière de décision est définie par :
$$\sum_{i=1}^{n} w_i x_i + w_0 = 0$$

C'est un **hyperplan** dans l'espace $\mathbb{R}^n$.

**Exemple en 2D :** Pour $(w_1, w_2) = (1, 1)$ et $w_0 = -1.5$ :

L'hyperplan est : $x_1 + x_2 - 1.5 = 0$

Soit : $x_2 = -x_1 + 1.5$

---

## 4. Fonction de Coût

### 4.1 Définition

La fonction de coût (ou fonction de perte) mesure l'écart entre les prédictions du modèle et les valeurs réelles.

Pour un exemple d'entraînement $(x, y^d)$ où :
- $x$ : vecteur d'entrée
- $y^d$ : sortie désirée (réelle)
- $\hat{y}$ : sortie prédite

### 4.2 Erreur Quadratique (pour un exemple)

$$E = \frac{1}{2}(y^d - \hat{y})^2$$

Le facteur $\frac{1}{2}$ simplifie les dérivées.

### 4.3 Erreur Quadratique Moyenne (MSE)

Pour un ensemble de $m$ exemples d'entraînement :

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (y_i^d - \hat{y}_i)^2$$

où $J(w)$ dépend des poids $w$ du modèle.

### 4.4 Propriétés de la Fonction de Coût

**Objectif :** Minimiser $J(w)$ pour trouver les meilleurs poids.

**Interprétation géométrique :**
- $J(w)$ définit une surface dans l'espace des poids
- Le minimum global correspond aux meilleurs poids possibles

---

## 5. Gradient d'une Fonction

### 5.1 Définition

Le **gradient** d'une fonction $f$ de $n$ variables est le vecteur de ses dérivées partielles :

$$\nabla f(x_1, x_2, \ldots, x_n) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

### 5.2 Interprétation Géométrique

Le gradient indique :
- **La direction de la variation la plus rapide** de la fonction
- **Le vecteur pointe vers le maximum local**

### 5.3 Propriété Fondamentale

Pour minimiser $f$, on se déplace dans la **direction opposée au gradient** :

$$x_{\text{nouveau}} = x_{\text{ancien}} - \alpha \nabla f(x_{\text{ancien}})$$

où $\alpha > 0$ est le **taux d'apprentissage**.

---

## 6. Dérivation de la Règle Delta

### 6.1 Contexte

La **règle delta** permet d'ajuster les poids d'un perceptron pour minimiser l'erreur de prédiction.

### 6.2 Fonction de Coût pour un Exemple

Pour un exemple $(x, y^d)$, la sortie prédite est :

$$\hat{y} = f\left(\sum_{i=1}^{n} x_i w_i + w_0\right)$$

L'erreur est :
$$E = \frac{1}{2}(y^d - \hat{y})^2$$

### 6.3 Objectif

Calculer $\frac{\partial E}{\partial w_j}$ pour chaque poids $w_j$.

### 6.4 Dérivation avec la Règle de la Chaîne

Posons :
- $z = \sum_{i=1}^{n} x_i w_i + w_0$ (somme pondérée)
- $\hat{y} = f(z)$ (sortie après activation)

Alors :
$$\frac{\partial E}{\partial w_j} = \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$

**Calcul de chaque terme :**

1. **Dérivée par rapport à $\hat{y}$ :**
   $$\frac{\partial E}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}}\left[\frac{1}{2}(y^d - \hat{y})^2\right] = -(y^d - \hat{y})$$

2. **Dérivée de la fonction d'activation :**
   $$\frac{\partial \hat{y}}{\partial z} = f'(z)$$

3. **Dérivée de la somme pondérée :**
   $$\frac{\partial z}{\partial w_j} = x_j$$

### 6.5 Résultat Final

$$\frac{\partial E}{\partial w_j} = -(y^d - \hat{y}) \cdot f'(z) \cdot x_j$$

On définit le **terme d'erreur** :
$$\delta = (y^d - \hat{y}) \cdot f'(z)$$

Donc :
$$\boxed{\frac{\partial E}{\partial w_j} = -\delta \cdot x_j}$$

### 6.6 Règle de Mise à Jour des Poids

Pour minimiser l'erreur, on ajuste les poids dans la direction opposée au gradient :

$$w_j^{\text{nouveau}} = w_j^{\text{ancien}} - \eta \frac{\partial E}{\partial w_j}$$

Soit :
$$\boxed{w_j^{\text{nouveau}} = w_j^{\text{ancien}} + \eta \cdot \delta \cdot x_j}$$

où $\eta$ est le **taux d'apprentissage**.

### 6.7 Cas Particulier : Fonction d'Activation Linéaire

Si $f(z) = z$ (pas d'activation), alors $f'(z) = 1$ et :

$$\delta = y^d - \hat{y}$$

La règle devient :
$$\boxed{\Delta w_j = \eta (y^d - \hat{y}) x_j}$$

**C'est la règle delta originale (règle de Widrow-Hoff).**

---

## 7. Algorithme de Descente de Gradient

### 7.1 Principe Général

L'algorithme de **descente de gradient** cherche le minimum d'une fonction de coût $J(w)$ en itérant :

$$w^{(t+1)} = w^{(t)} - \eta \nabla J(w^{(t)})$$

où :
- $w^{(t)}$ : vecteur de poids à l'itération $t$
- $\eta$ : taux d'apprentissage
- $\nabla J(w^{(t)})$ : gradient de la fonction de coût

### 7.2 Algorithme Détaillé

**Entrée :** Ensemble d'entraînement $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$

**Initialisation :** $w^{(0)}$ (aléatoire ou à zéro)

**Répéter jusqu'à convergence :**

1. Calculer le gradient :
   $$\nabla J(w) = \frac{1}{m} \sum_{i=1}^{m} \nabla E_i(w)$$
   
   où $E_i$ est l'erreur sur l'exemple $i$.

2. Mettre à jour les poids :
   $$w := w - \eta \nabla J(w)$$

**Critère d'arrêt :**
- Nombre maximal d'itérations atteint
- $\|\nabla J(w)\| < \epsilon$ (convergence)
- Changement de $J(w)$ très faible

### 7.3 Variantes de la Descente de Gradient

#### 7.3.1 Descente de Gradient par Lot (Batch Gradient Descent)

On calcule le gradient sur **tout l'ensemble d'entraînement** à chaque itération :

$$\nabla J(w) = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial E_i}{\partial w}$$

**Avantages :** Convergence stable
**Inconvénients :** Coût de calcul élevé pour grands ensembles

#### 7.3.2 Descente de Gradient Stochastique (SGD)

On met à jour les poids après **chaque exemple** :

$$w := w - \eta \frac{\partial E_i}{\partial w}$$

**Avantages :** Convergence rapide, peut échapper aux minima locaux
**Inconvénients :** Trajectoire bruitée

#### 7.3.3 Mini-Batch Gradient Descent

Compromis : on utilise un **petit sous-ensemble** (mini-batch) à chaque itération.

**Taille typique de mini-batch :** 32, 64, 128, 256

### 7.4 Choix du Taux d'Apprentissage

Le taux $\eta$ est crucial :

- **$\eta$ trop petit :** Convergence très lente
- **$\eta$ trop grand :** Divergence, oscillations

**Stratégies :**
- Taux fixe
- Taux décroissant : $\eta_t = \frac{\eta_0}{1 + \text{decay} \cdot t}$
- Recherche par validation croisée

---

## 8. Propriétés de Convexité

### 8.1 Définitions

#### 8.1.1 Fonction Convexe

Une fonction $f: \mathbb{R}^n \to \mathbb{R}$ est **convexe** si pour tous $x, y \in \mathbb{R}^n$ et $\lambda \in [0, 1]$ :

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

**Interprétation géométrique :** Le segment reliant deux points de la courbe est au-dessus de la courbe.

#### 8.1.2 Fonction Strictement Convexe

Si l'inégalité est stricte pour $\lambda \in (0, 1)$ et $x \neq y$ :

$$f(\lambda x + (1-\lambda)y) < \lambda f(x) + (1-\lambda) f(y)$$

### 8.2 Critère du Hessien

Pour une fonction deux fois dérivable, $f$ est convexe si et seulement si sa **matrice hessienne** $H$ est semi-définie positive :

$$H = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix} \succeq 0$$

### 8.3 Propriétés Importantes pour l'Optimisation

#### 8.3.1 Fonction Convexe

**Si la fonction d'erreur est convexe :**

✅ **Le minimum est unique**

✅ **Tout minimum local est un minimum global**

✅ **L'algorithme de descente de gradient converge vers la solution optimale**

**Exemple :** Erreur quadratique moyenne (MSE) avec modèle linéaire.

#### 8.3.2 Fonction Non-Convexe

**Si la fonction d'erreur est non-convexe :**

❌ **Plusieurs minima locaux peuvent exister**

❌ **Le résultat dépend de l'initialisation des poids**

❌ **Risque de convergence vers un minimum local non optimal**

**Exemple :** Réseaux de neurones profonds avec fonctions d'activation non-linéaires.

### 8.4 Analyse de la Convexité de l'Erreur Quadratique

Pour un perceptron avec activation linéaire :

$$\hat{y} = w^T x$$

L'erreur quadratique est :

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (y_i^d - w^T x_i)^2$$

**Calcul du Hessien :**

$$\frac{\partial^2 J}{\partial w_k \partial w_j} = \frac{1}{m} \sum_{i=1}^{m} x_{ik} x_{ij}$$

Soit en forme matricielle :

$$H = \frac{1}{m} X^T X$$

où $X$ est la matrice des données.

**Conclusion :** $H$ est semi-définie positive (car $X^T X \succeq 0$), donc $J(w)$ est **convexe**.

### 8.5 Visualisation

```
Fonction Convexe :                  Fonction Non-Convexe :
      
        J(w)                               J(w)
         |                                  |
         |        __                        |      __        __
         |      /    \                      |    /    \    /    \
         |    /        \                    |  /        \/        \
         |__/            \                  |/                      \
         |________________                  |_______________________
              w* (unique)                    w₁*    w₂*   w₃* (multiples)
```

### 8.6 Implications Pratiques

**Pour le Perceptron (modèle linéaire) :**
- Fonction de coût convexe
- Solution optimale garantie
- Convergence stable

**Pour les Réseaux de Neurones Profonds :**
- Fonction de coût non-convexe
- Multiples minima locaux
- Nécessite des techniques avancées :
  - Bonnes initialisations (Xavier, He)
  - Optimisateurs adaptatifs (Adam, RMSprop)
  - Régularisation

---

## Résumé des Formules Clés

| Concept | Formule |
|---------|---------|
| Sigmoïde | $\sigma(x) = \frac{1}{1+e^{-x}}$ |
| Dérivée Sigmoïde | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ |
| Dérivée Tanh | $\tanh'(x) = 1 - \tanh^2(x)$ |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ |
| Erreur Quadratique | $E = \frac{1}{2}(y^d - \hat{y})^2$ |
| Gradient | $\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)$ |
| Règle Delta | $\Delta w_j = \eta (y^d - \hat{y}) f'(z) x_j$ |
| Descente de Gradient | $w := w - \eta \nabla J(w)$ |

---

**Fin des Dérivations - Chapitre 1**