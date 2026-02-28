# FICHE EXAM — Chapitre 1 : Priorités et Prédictions

---

## ⭐⭐⭐ QUASI-CERTAINS À L'EXAMEN

### 1. Algorithme du perceptron complet (Hebb ou règle delta)
**Ce qu'on vous donnera** : une base d'apprentissage avec 2 à 4 exemples, des conditions initiales ($w_1=w_2=\Theta=0$, $\beta=1$), et on vous demande de déterminer les poids.

**Ce qu'il faut faire** :
1. Init : $w_1=w_2=\ldots=\Theta=0$, $\beta$ donné
2. Pour chaque exemple $(X_i, y_i^d)$ :
   - Calculer $S = \sum x_k w_k$
   - Calculer $\hat{y}_i = \text{signe}(S - \Theta)$ → $S-\Theta > 0 \Rightarrow +1$, sinon $-1$
   - Si $y_i^d \neq \hat{y}_i$ : $w_k \leftarrow w_k + \beta(x_k y_i^d)$
3. Recommencer jusqu'à classification correcte de tous les exemples

**Piège** : Ne pas oublier de revenir au début de la base si on a fait des corrections !

---

### 2. Règle delta — Application numérique
**Ce qu'on vous donnera** : entrées, sorties désirées, poids initiaux, $\beta$.

**Formules clés** :
$$\hat{y} = f(w \cdot x + w_0) \qquad \text{Err} = y^d - \hat{y}$$
$$w_i \leftarrow w_i + \beta x_i \cdot \text{Err} \qquad w_0 \leftarrow w_0 + \beta \cdot \text{Err}$$

---

### 3. Neurone McCulloch-Pitts — Portes logiques ET/OU/NON
**Ce qu'on vous donnera** : on demande de vérifier ou construire un neurone réalisant une porte logique.

**À mémoriser** :
- Poids $w_1=w_2=1$, $\theta=1$ → **OU**
- Poids $w_1=w_2=1$, $\theta=2$ → **ET**
- Poids $w_1=-1$, $\theta=0$ (ou $w_0=0.5$) → **NON**

---

### 4. Les 4 fonctions d'activation — Formules et avantages
**Ce qu'on vous demandera** : donner la formule, tracer l'allure, citer les avantages.

| Fonction | Formule | Avantage clé |
|----------|---------|--------------|
| Heaviside | $H(x)=0$ si $x<0$, $1$ si $x\geq 0$ | Simple, modèle original |
| Sigmoïde | $\frac{1}{1+e^{-x}}$ | Continue, dérivable |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | Sortie centrée en 0 |
| **ReLU** | $\max(0,x)$ | **Évite la disparition du gradient** |

---

### 5. Descente de gradient — 2 à 3 itérations numériques
**Exemple type** : $f(x) = (x+1)^2 - 2$, $x_0=-4$, $\beta=0.1$

**Étapes** :
1. $\nabla f(x) = 2(x+1)$
2. $x_1 = x_0 - \beta \nabla f(x_0) = -4 - 0.1 \times (-6) = -4 + 0.6 = -3.4$
3. $x_2 = -3.4 - 0.1 \times (-4.8) = -3.4 + 0.48 = -2.92$
4. Convergence vers $x^* = -1$

---

### 6. Calcul de la MSE
**Formule** : $\text{Loss} = \frac{1}{N} \sum_{i=1}^N (y_i^d - \hat{y}_i)^2$

**Exemple** : $Y^d=(1,0,0,0)$, $\hat{Y}=(0.8,0.2,0.1,0.7)$
$$\text{Loss} = \frac{1}{4}[0.04 + 0.04 + 0.01 + 0.49] = \frac{0.58}{4} = 0.145$$

---

## ⭐⭐ PROBABLES À L'EXAMEN

### 7. Supervisé vs Non supervisé — Différences
- Supervisé : données **étiquetées**, $B=(X_i,y_i^d)$, objectif = apprendre $f(X)\approx y$
- Non supervisé : données **non étiquetées**, $B=(X_i)$, objectif = trouver des structures
- Tâches non supervisées : clustering, réduction de dimension, détection d'anomalies

### 8. Classification vs Régression
- Classification : $y^d \in \{1,\ldots,C\}$ (catégorie)
- Régression : $y^d \in \mathbb{R}$ (valeur numérique)

### 9. Algorithme de Hebb — Limitation
- La loi de Hebb ne converge **pas toujours**, même si le problème est linéairement séparable
- Solution alternative : la **règle delta** qui utilise l'erreur

### 10. Convexité et convergence
- Convexe → minimum unique, convergence certaine
- Non-convexe → minima locaux, résultat dépend du point de départ
- Même $w$ rencontré deux fois → données **non séparables**

### 11. Choix de la fonction de perte selon le problème
- Régression → **MSE** : $(y^d - \hat{y})^2$
- Classification binaire → **log-loss** : $-y^d\log\hat{y} - (1-y^d)\log(1-\hat{y})$
- Multiclasse → **entropie croisée** : $-\sum_k y_k^d \log \hat{y}_k$

### 12. Gradient d'une fonction — Calcul
- $f(x) = (x+1)^2 - 2$ → $\nabla f = 2(x+1)$
- $g(x_1,x_2) = \frac{1}{2}(x_1-1)^2 + \frac{1}{2}(x_2-2)^2$ → $\nabla g = (x_1-1, x_2-2)$

---

## ⭐ POSSIBLES À L'EXAMEN

### 13. Neurone biologique — 4 composants
Dendrites (entrées) → Synapses (transmission) → Axone (sortie) → Noyau (activation)

### 14. Tableau de coactivation de Hebb
Connexion renforcée uniquement si $x_i=1$ ET $x_j=1$. Sinon $\Delta w = 0$.

### 15. Les 3 variantes de la descente de gradient
Batch / Stochastique / Mini-batch (avantages et inconvénients)

### 16. Effets du taux d'apprentissage $\beta$
- Trop grand → oscillations autour du minimum
- Trop petit → convergence lente
- En pratique → diminué progressivement

### 17. Frontière de décision — Exemple numérique
$(w_1,w_2)=(1,1)$, $w_0=-1.5$ → frontière $x_1+x_2-1.5=0$ → réalise AND

### 18. Séparabilité linéaire
- Problèmes séparables : AND, OR, NOT
- Problème NON séparable : **XOR** → nécessite MLP

---

## FORMULES À AVOIR PAR CŒUR

```
Base supervisée    : B = (Xᵢ, yᵢᵈ)₁≤ᵢ≤ₙ
Neurone formel     : S = Σxᵢwᵢ,  y=1 si S≥θ, 0 sinon
Sigmoïde           : σ(x) = 1/(1+e⁻ˣ)
ReLU               : f(x) = max(0,x)
Règle delta        : wᵢ ← wᵢ + βxᵢ(yᵈ-ŷ)
Mise à jour biais  : w₀ ← w₀ + β(yᵈ-ŷ)
Loi Hebb           : Δw(i,j) = β(xᵢxⱼ)
Risque empirique   : E(w) = (1/N)Σ Errᵢ(w)
MSE                : Errᵢ = (yᵈ-ŷ)²
Gradient           : ∇f = (∂f/∂x₁, ..., ∂f/∂xₙ)
Descente gradient  : xₜ = xₜ₋₁ - β∇f(xₜ₋₁)
Frontière          : Σwᵢxᵢ + w₀ = 0
```

---

## ERREURS FRÉQUENTES À ÉVITER

1. **Oublier le biais** $w_0$ dans la mise à jour de la règle delta
2. **Confondre loi de Hebb** (utilise $x_i y_i^d$) et **règle delta** (utilise $x_i \text{Err}_i$)
3. **Signe dans la descente** : $x_t = x_{t-1} \mathbf{-} \beta\nabla f$ (signe **moins** !)
4. **Epoch ≠ mise à jour** : une epoch = passage complet sur TOUTE la base
5. **Convergence de Hebb** : ne pas supposer qu'elle converge toujours
6. **XOR n'est pas linéairement séparable** → perceptron simple ne peut pas l'apprendre
