# Mathematical Cheatsheet - Neural Networks Chapter 1

## 📐 Quick Reference Guide

---

## 1. NEURONE FORMEL (McCulloch-Pitts, 1943)

### Entrées et Sortie
- Entrées : $X = [x_1, x_2, \ldots, x_n]^T$ où $x_i \in \{0, 1\}$
- Sortie : $y \in \{0, 1\}$

### Somme Pondérée
$$S = \sum_{i=1}^{n} x_i w_i$$

### Activation (Heaviside)
$$y = H(S - \theta) = \begin{cases}
1 & \text{si } S \geq \theta \\
0 & \text{sinon}
\end{cases}$$

---

## 2. FONCTIONS D'ACTIVATION

### Heaviside (Step Function)
$$H(x) = \begin{cases}
0 & \text{si } x < 0 \\
1 & \text{si } x \geq 0
\end{cases}$$

### Sigmoïde
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Propriétés :**
- Sortie : $(0, 1)$
- Lisse et dérivable
- Symétrique autour de $x = 0$

### Tanh (Tangente Hyperbolique)
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
$$\tanh'(x) = 1 - \tanh^2(x)$$

**Propriétés :**
- Sortie : $(-1, 1)$
- Centrée sur 0
- $\tanh(x) = 2\sigma(2x) - 1$

### ReLU (Rectified Linear Unit)
$$\text{ReLU}(x) = \max(0, x) = \begin{cases}
0 & \text{si } x < 0 \\
x & \text{si } x \geq 0
\end{cases}$$
$$\text{ReLU}'(x) = \begin{cases}
0 & \text{si } x < 0 \\
1 & \text{si } x > 0
\end{cases}$$

**Avantages :**
- Calcul rapide
- Évite la disparition du gradient
- **Fonction la plus utilisée en DL moderne**

---

## 3. PERCEPTRON

### Modèle Général
$$\hat{y} = f\left(\sum_{i=0}^{n} x_i w_i\right) = f(w^T x)$$

où $x_0 = 1$ (biais intégré)

### Forme Explicite avec Biais
$$\hat{y} = f\left(\sum_{i=1}^{n} x_i w_i + w_0\right)$$

### Classifieur Binaire
$$\text{signe}(x) = \begin{cases}
1 & \text{si } w^T x + w_0 > 0 \\
-1 & \text{sinon}
\end{cases}$$

### Hyperplan de Décision
$$\sum_{i=1}^{n} w_i x_i + w_0 = 0$$

**En 2D :** $w_1 x_1 + w_2 x_2 + w_0 = 0$

---

## 4. FONCTIONS DE COÛT

### Erreur Quadratique (1 exemple)
$$E = \frac{1}{2}(y^d - \hat{y})^2$$

### MSE (Mean Squared Error)
$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (y_i^d - \hat{y}_i)^2$$

### Erreur Absolue (MAE)
$$J(w) = \frac{1}{m} \sum_{i=1}^{m} |y_i^d - \hat{y}_i|$$

---

## 5. GRADIENT

### Définition
$$\nabla f(x_1, \ldots, x_n) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

### Propriété Clé
Le gradient pointe dans la **direction de plus grande augmentation** de $f$.

Pour minimiser : se déplacer dans la **direction opposée** : $-\nabla f$

---

## 6. RÈGLE DELTA

### Dérivation Complète

**Fonction de coût :**
$$E = \frac{1}{2}(y^d - \hat{y})^2$$

**Sortie du neurone :**
$$\hat{y} = f(z)$$ où $$z = \sum_{i=1}^{n} x_i w_i + w_0$$

**Gradient par rapport à $w_j$ :**
$$\frac{\partial E}{\partial w_j} = \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$

$$= -(y^d - \hat{y}) \cdot f'(z) \cdot x_j$$

**Terme d'erreur :**
$$\delta = (y^d - \hat{y}) \cdot f'(z)$$

**Mise à jour des poids :**
$$\boxed{w_j := w_j + \eta \cdot \delta \cdot x_j}$$

### Cas Activation Linéaire
Si $f(z) = z$, alors $f'(z) = 1$ :

$$\boxed{\Delta w_j = \eta (y^d - \hat{y}) x_j}$$

---

## 7. DESCENTE DE GRADIENT

### Algorithme Général
$$w^{(t+1)} = w^{(t)} - \eta \nabla J(w^{(t)})$$

### Forme Composante par Composante
$$w_j := w_j - \eta \frac{\partial J}{\partial w_j}$$

### Gradient pour MSE (Batch)
$$\nabla J(w) = \frac{1}{m} \sum_{i=1}^{m} -(y_i^d - \hat{y}_i) f'(z_i) x_i$$

### Variantes

| Type | Formule | Utilisation |
|------|---------|-------------|
| **Batch GD** | $\nabla J = \frac{1}{m}\sum_{i=1}^{m} \nabla E_i$ | Tout le dataset |
| **Stochastic GD** | $w := w - \eta \nabla E_i$ | Un exemple à la fois |
| **Mini-Batch GD** | $\nabla J = \frac{1}{b}\sum_{i=1}^{b} \nabla E_i$ | Sous-ensemble de taille $b$ |

---

## 8. LOI DE HEBB

### Principe
**"Neurons that fire together, wire together"**

Si deux neurones sont **activés simultanément**, leur connexion est **renforcée**.

### Règle de Mise à Jour
$$\Delta w_{ij} = \eta \cdot a_i \cdot a_j$$

où :
- $a_i$ : activation du neurone pré-synaptique
- $a_j$ : activation du neurone post-synaptique
- $\eta$ : taux d'apprentissage

### Forme Générale
Si $a_i$ et $a_j$ sont simultanément actifs → $w_{ij}$ **augmente**

Si activation asynchrone → $w_{ij}$ **diminue**

---

## 9. CONVEXITÉ

### Définition - Fonction Convexe
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

pour tout $x, y$ et $\lambda \in [0, 1]$

### Critère du Hessien
$f$ est convexe ⟺ Hessien $H \succeq 0$ (semi-définie positive)

$$H = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

### Propriétés pour l'Optimisation

| Fonction Convexe ✅ | Fonction Non-Convexe ❌ |
|---------------------|------------------------|
| Minimum unique | Multiples minima locaux |
| Min local = Min global | Min local ≠ Min global |
| Convergence garantie | Dépend de l'initialisation |
| Ex: MSE linéaire | Ex: Réseaux profonds |

---

## 10. CALCULS PRATIQUES

### Dérivées Partielles Courantes

$$\frac{\partial}{\partial w}(w^T x) = x$$

$$\frac{\partial}{\partial w}\left(\frac{1}{2}\|y - \hat{y}\|^2\right) = -(y - \hat{y})\frac{\partial \hat{y}}{\partial w}$$

$$\frac{\partial}{\partial w}(wx + b) = x$$

### Règle de la Chaîne
$$\frac{\partial f(g(x))}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$$

### Exemple : Neurone avec Sigmoïde
Si $\hat{y} = \sigma(w^T x)$ :

$$\frac{\partial \hat{y}}{\partial w} = \sigma(w^T x)(1 - \sigma(w^T x)) \cdot x = \hat{y}(1 - \hat{y}) \cdot x$$

---

## 11. OPÉRATIONS LOGIQUES

### Avec Neurone Formel (seuil $\theta$)

**Opération OU :** $\theta = 1$, $w_i = 1$
- Sortie = 1 si au moins une entrée vaut 1

**Opération ET :** $\theta = 2$, $w_i = 1$  
- Sortie = 1 uniquement si toutes les entrées valent 1

**Opération NON :** $\theta = 0$, $w = -1$
- Sortie = inverse de l'entrée

---

## 12. HYPERPARAMÈTRES IMPORTANTS

| Paramètre | Symbole | Rôle | Valeurs Typiques |
|-----------|---------|------|------------------|
| Taux d'apprentissage | $\eta$ | Taille du pas | $10^{-4}$ à $10^{-1}$ |
| Nombre d'époques | $T$ | Itérations complètes | 10 à 1000+ |
| Taille mini-batch | $b$ | Exemples par mise à jour | 32, 64, 128, 256 |
| Seuil d'activation | $\theta$ | Décision binaire | Varie selon problème |

---

## 13. FORMULES MATRICIELLES

### Pour un Batch d'Exemples

**Données :**
$$X = \begin{pmatrix}
x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{pmatrix}_{m \times n}$$

**Prédictions (vectorisées) :**
$$\hat{Y} = f(XW + b)$$

où :
- $W$ : vecteur de poids $(n \times 1)$
- $b$ : biais (scalaire ou vecteur)
- $f$ : fonction d'activation appliquée élément par élément

**Gradient (forme matricielle) :**
$$\nabla_W J = \frac{1}{m} X^T (Y - \hat{Y}) \odot f'(Z)$$

où $\odot$ est le produit élément par élément (Hadamard).

---

## 🎯 POINTS CLÉS À RETENIR

1. **Sigmoïde dérivée** : $\sigma'(x) = \sigma(x)(1-\sigma(x))$ ← Très utilisé en examen!

2. **Règle delta** : $\Delta w = \eta (y^d - \hat{y}) f'(z) x$ ← Fondamental

3. **Descente de gradient** : $w := w - \eta \nabla J(w)$ ← Cœur de l'apprentissage

4. **ReLU** : Fonction d'activation la plus courante en DL moderne

5. **Convexité** : MSE linéaire = convexe ✅ / Réseaux profonds = non-convexe ❌

6. **Loi de Hebb** : "Fire together, wire together"

7. **Hyperplan** : $\sum w_i x_i + w_0 = 0$ sépare les classes

---

## 📊 COMPARAISON DES FONCTIONS D'ACTIVATION

| Fonction | Formule | Dérivée | Sortie | Avantages | Inconvénients |
|----------|---------|---------|--------|-----------|---------------|
| Heaviside | $H(x)$ | Non définie | $\{0,1\}$ | Simple | Non dérivable |
| Sigmoïde | $\frac{1}{1+e^{-x}}$ | $\sigma(1-\sigma)$ | $(0,1)$ | Lisse, probabilités | Disparition gradient |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-\tanh^2$ | $(-1,1)$ | Centrée sur 0 | Disparition gradient |
| ReLU | $\max(0,x)$ | $\mathbb{1}_{x>0}$ | $[0,\infty)$ | Rapide, pas de disparition | Neurones morts |

---

**Dernière mise à jour :** Chapitre 1 - Concepts Fondamentaux