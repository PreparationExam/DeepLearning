# RÉSUMÉ COURT — Chapitre 1 : Ce que tu DOIS savoir

---

## 1. HIÉRARCHIE

```
IA ⊃ ML ⊃ DL
```
- **ML** = détection automatique de régularités dans les données
- **DL** = ML utilisant des réseaux de neurones interconnectés
- **Supervisé** = données étiquetées $B=(X_i, y_i^d)_{1 \leq i \leq N}$
- **Non supervisé** = données non étiquetées $B=(X_i)_{1 \leq i \leq N}$
- **Classification** : $y \in \{1,\ldots,C\}$ | **Régression** : $y \in \mathbb{R}$

---

## 2. NEURONE FORMEL (McCulloch & Pitts, 1943)

**Deux phases :**
1. Agrégation : $S = \sum_{i=1}^n x_i w_i$
2. Activation (Heaviside) : $y = 1$ si $S \geq \theta$, sinon $y = 0$

**Entrées et sorties binaires** $\in \{0, 1\}$. Pas d'apprentissage automatique.

**Portes logiques** (poids = 1) :
- $\theta = 1$ → **OU**
- $\theta = 2$ → **ET**

---

## 3. FONCTIONS D'ACTIVATION

| Fonction | Formule | Sortie | Usage |
|----------|---------|--------|-------|
| Heaviside | $H(x) = 0$ si $x<0$, $1$ si $x \geq 0$ | $\{0,1\}$ | Neurone formel |
| Sigmoïde | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $]0,1[$ | Classification binaire |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $]-1,1[$ | Données centrées |
| **ReLU** | $\max(0,x)$ | $[0,+\infty[$ | **DL moderne** |

**ReLU** = la plus utilisée. Limite la disparition du gradient.

---

## 4. PERCEPTRON (Rosenblatt, 1957)

- Classification binaire, apprentissage supervisé
- $\hat{y} = f\left(\sum_{i=1}^n x_i w_i + w_0\right)$
- Biais : $x_0 = 1$, $w_0 = -\Theta$
- **Frontière de décision** (hyperplan) : $\sum_{i=1}^n w_i x_i + w_0 = 0$
- **Limite** : problèmes linéairement séparables uniquement

---

## 5. LOI DE HEBB

$$\Delta w(i,j) = \beta(x_i x_j)$$

Connexion renforcée **seulement si les deux neurones sont actifs** ($x_i=1$ ET $x_j=1$).
**Limitation** : ne converge pas toujours.

---

## 6. RÈGLE DELTA

$$w_i \leftarrow w_i + \beta x_i (y_i^d - \hat{y}_i)$$
$$w_0 \leftarrow w_0 + \beta (y_i^d - \hat{y}_i)$$

- Err = 0 → poids inchangé
- Err > 0 → poids augmente
- Err < 0 → poids diminue

---

## 7. FONCTIONS DE PERTE

| Problème | Formule |
|----------|---------|
| Régression (MSE) | $(y_i^d - \hat{y}_i)^2$ |
| Classification binaire (log-loss) | $-y_i^d \log(\hat{y}_i) - (1-y_i^d)\log(1-\hat{y}_i)$ |
| Multiclasse (entropie croisée) | $-\sum_{k=1}^K y_k^d \log(\hat{y}_k)$ |

**Risque empirique** : $E(w) = \frac{1}{N}\sum_{i=1}^N \text{Err}_i(w)$

---

## 8. GRADIENT ET DESCENTE

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

Gradient = direction de **montée** maximale → pour minimiser : aller dans la **direction opposée**.

**Descente de gradient :**
$$x_t = x_{t-1} - \beta \nabla f(x_{t-1})$$

**3 variantes :**
- **Batch** : correction après tous les exemples (stable, lent)
- **Stochastique** : correction après 1 exemple (rapide, irrégulier)
- **Mini-batch** : correction après un sous-ensemble (compromis ✓)

---

## 9. CONVERGENCE

- **Convexe** → minimum unique, convergence garantie
- **Non-convexe** → minima locaux, dépend du point de départ
- Si même $w$ rencontré 2 fois → données **non linéairement séparables**
- Borne max itérations : $(N+1)^2 \cdot 2^{(N+1)\log(N+1)}$
- $\beta$ trop grand → oscillations | $\beta$ trop petit → lent
