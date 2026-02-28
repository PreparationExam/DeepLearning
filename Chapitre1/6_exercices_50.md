# 50 EXERCICES ET PROBLÈMES AVEC SOLUTIONS — Chapitre 1

---

# BLOC A — QCM ET VRAI/FAUX (Exercices 1–10)

---

**Exercice 1 (QCM)** — Quelle affirmation est correcte ?
- (a) Le Deep Learning est une sous-catégorie du Machine Learning
- (b) Le Machine Learning est une sous-catégorie du Deep Learning
- (c) IA et ML sont la même chose
- (d) Le DL n'utilise pas de réseaux de neurones

**Solution :**
✅ **(a)** Le Deep Learning est une **sous-catégorie du ML**, qui est lui-même une sous-catégorie de l'IA. La hiérarchie est : IA ⊃ ML ⊃ DL.

---

**Exercice 2 (QCM)** — Laquelle de ces tâches est un problème de **classification** ?
- (a) Prédire la température demain
- (b) Estimer le prix d'un appartement
- (c) Déterminer si un email est un spam
- (d) Prédire la tension artérielle d'un patient

**Solution :**
✅ **(c)** Déterminer si un email est un spam est une classification binaire : $y \in \{\text{spam}, \text{non-spam}\}$. Les autres sont des régressions car $y \in \mathbb{R}$.

---

**Exercice 3 (QCM)** — Quelle fonction d'activation est la **plus utilisée** dans le Deep Learning moderne ?
- (a) Heaviside
- (b) Sigmoïde
- (c) Tanh
- (d) ReLU

**Solution :**
✅ **(d) ReLU** — $f(x) = \max(0,x)$. Elle est la plus utilisée car elle combine simplicité et performance, et limite le problème de disparition du gradient rencontré avec la sigmoïde.

---

**Exercice 4 (Vrai/Faux)** — La loi de Hebb converge toujours si le problème est linéairement séparable.

**Solution :**
❌ **FAUX.** La loi de Hebb ne fonctionne **pas toujours**, même lorsque le problème est linéairement séparable. C'est sa principale limitation. L'exercice 2 du cours le démontre : une solution existe mais Hebb ne la trouve pas.

---

**Exercice 5 (Vrai/Faux)** — Le perceptron peut apprendre la fonction XOR.

**Solution :**
❌ **FAUX.** XOR n'est **pas linéairement séparable** : il est impossible de séparer les points $(0,0)$, $(1,1)$ (classe 0) des points $(0,1)$, $(1,0)$ (classe 1) avec un seul hyperplan. Le perceptron ne peut apprendre que des problèmes linéairement séparables.

---

**Exercice 6 (QCM)** — Dans la règle delta, si la prédiction $\hat{y}$ est trop faible (inférieure à la sortie désirée $y^d$), que se passe-t-il avec le poids $w_i$ (en supposant $x_i > 0$) ?
- (a) $w_i$ diminue
- (b) $w_i$ ne change pas
- (c) $w_i$ augmente
- (d) $w_i$ est remis à zéro

**Solution :**
✅ **(c)** Si $\hat{y} < y^d$, alors $\text{Err} = y^d - \hat{y} > 0$. La mise à jour est $w_i \leftarrow w_i + \beta x_i \text{Err}$. Avec $x_i > 0$ et $\text{Err} > 0$, la correction $\beta x_i \text{Err} > 0$, donc $w_i$ **augmente**.

---

**Exercice 7 (QCM)** — Quelle est l'expression du risque empirique ?
- (a) $E(w) = \sum_{i=1}^N \text{Err}_i(w)$
- (b) $E(w) = \frac{1}{N} \sum_{i=1}^N \text{Err}_i(w)$
- (c) $E(w) = \max_i \text{Err}_i(w)$
- (d) $E(w) = \frac{1}{N^2} \sum_{i=1}^N \text{Err}_i(w)$

**Solution :**
✅ **(b)** $E(w) = \frac{1}{N} \sum_{i=1}^N \text{Err}_i(w)$. C'est la **moyenne** des erreurs sur tous les exemples, pas leur somme.

---

**Exercice 8 (Vrai/Faux)** — Dans la descente de gradient, on met à jour les paramètres dans la **direction** du gradient.

**Solution :**
❌ **FAUX.** On met à jour dans la **direction opposée** au gradient : $x_t = x_{t-1} - \beta \nabla f(x_{t-1})$. Le gradient pointe vers la montée maximale ; pour minimiser, on va dans le sens contraire.

---

**Exercice 9 (QCM)** — Une epoch est :
- (a) Une mise à jour de poids pour un seul exemple
- (b) Une itération complète sur l'ensemble des exemples d'apprentissage
- (c) La valeur du taux d'apprentissage
- (d) Le nombre total d'exemples

**Solution :**
✅ **(b)** Une **epoch** correspond à une **itération complète** sur l'ensemble des exemples d'apprentissage. Après plusieurs epochs, les poids convergent progressivement.

---

**Exercice 10 (Vrai/Faux)** — La sigmoïde est symétrique par rapport à l'origine.

**Solution :**
❌ **FAUX.** C'est le **tanh** qui est une fonction impaire, donc symétrique par rapport à l'origine. La sigmoïde $\sigma(x) = \frac{1}{1+e^{-x}}$ a une sortie dans $]0,1[$ avec $\sigma(0) = 0.5$, elle n'est pas symétrique par rapport à l'origine.

---

# BLOC B — CALCULS SUR LES FONCTIONS D'ACTIVATION (Exercices 11–18)

---

**Exercice 11** — Calculez la sortie d'un neurone formel avec $x_1=1$, $x_2=0$, $w_1=0.5$, $w_2=0.8$, $\theta=0.4$.

**Solution :**
1. Phase d'agrégation :
$$S = x_1 w_1 + x_2 w_2 = 1 \times 0.5 + 0 \times 0.8 = 0.5$$
2. Phase d'activation : $S = 0.5 \geq \theta = 0.4$
$$\boxed{y = 1}$$

---

**Exercice 12** — Calculez la sortie d'un neurone formel avec $x_1=1$, $x_2=1$, $w_1=w_2=1$, $\theta=2$. Quelle porte logique est réalisée ?

**Solution :**
$$S = 1 \times 1 + 1 \times 1 = 2 \geq \theta = 2 \Rightarrow y = 1$$
On peut vérifier les 4 cas :
- $(0,0)$: $S=0 < 2 \Rightarrow y=0$
- $(0,1)$: $S=1 < 2 \Rightarrow y=0$
- $(1,0)$: $S=1 < 2 \Rightarrow y=0$
- $(1,1)$: $S=2 \geq 2 \Rightarrow y=1$

Table de vérité identique à la **porte ET (AND)** ✓

---

**Exercice 13** — Calculez $\sigma(0)$, $\sigma(1)$, $\sigma(-1)$, $\sigma(2)$. (Sigmoïde)

**Solution :**
$$\sigma(x) = \frac{1}{1+e^{-x}}$$
- $\sigma(0) = \frac{1}{1+e^0} = \frac{1}{1+1} = \mathbf{0.5}$
- $\sigma(1) = \frac{1}{1+e^{-1}} = \frac{1}{1+0.368} \approx \mathbf{0.731}$
- $\sigma(-1) = \frac{1}{1+e^{1}} = \frac{1}{1+2.718} \approx \mathbf{0.269}$
- $\sigma(2) = \frac{1}{1+e^{-2}} = \frac{1}{1+0.135} \approx \mathbf{0.880}$

**Remarque** : $\sigma(-x) = 1 - \sigma(x)$, donc $\sigma(-1) = 1-\sigma(1) \approx 0.269$ ✓

---

**Exercice 14** — Calculez $\text{ReLU}(-3)$, $\text{ReLU}(0)$, $\text{ReLU}(2.5)$.

**Solution :**
$$f(x) = \max(0,x)$$
- $\text{ReLU}(-3) = \max(0,-3) = \mathbf{0}$
- $\text{ReLU}(0) = \max(0,0) = \mathbf{0}$
- $\text{ReLU}(2.5) = \max(0,2.5) = \mathbf{2.5}$

---

**Exercice 15** — Calculez $\tanh(0)$, $\tanh(1)$. Vérifiez que tanh est impaire.

**Solution :**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- $\tanh(0) = \frac{1-1}{1+1} = \frac{0}{2} = \mathbf{0}$
- $\tanh(1) = \frac{e - e^{-1}}{e + e^{-1}} = \frac{2.718 - 0.368}{2.718 + 0.368} = \frac{2.350}{3.086} \approx \mathbf{0.762}$

**Vérification impaire** : $\tanh(-x) = \frac{e^{-x}-e^x}{e^{-x}+e^x} = -\frac{e^x-e^{-x}}{e^x+e^{-x}} = -\tanh(x)$ ✓

---

**Exercice 16** — Un neurone a $x_1=0.5$, $x_2=-0.5$, $w_1=2$, $w_2=1$, $w_0=-0.5$. Calculez la sortie avec la fonction sigmoïde.

**Solution :**
1. Somme pondérée avec biais :
$$S = x_1 w_1 + x_2 w_2 + w_0 = 0.5 \times 2 + (-0.5) \times 1 + (-0.5) = 1 - 0.5 - 0.5 = 0$$
2. Application de la sigmoïde :
$$\hat{y} = \sigma(0) = 0.5$$
$$\boxed{\hat{y} = 0.5}$$

---

**Exercice 17** — Un perceptron implémente-t-il la porte OU avec $w_1=0.6$, $w_2=0.6$, $w_0=-0.5$ ? Vérifiez pour tous les cas.

**Solution :**
Frontière : $0.6x_1 + 0.6x_2 - 0.5 > 0 \Rightarrow$ sortie $+1$, sinon sortie $-1$ (ou 0/1 selon convention).

| $x_1$ | $x_2$ | $S = 0.6x_1+0.6x_2-0.5$ | Sortie | OU attendu |
|--------|--------|--------------------------|--------|------------|
| 0 | 0 | $-0.5$ | 0 | 0 ✓ |
| 0 | 1 | $0.1$ | 1 | 1 ✓ |
| 1 | 0 | $0.1$ | 1 | 1 ✓ |
| 1 | 1 | $0.7$ | 1 | 1 ✓ |

✅ Oui, ce perceptron réalise bien la porte OU.

---

**Exercice 18** — Concevez un neurone formel (choisissez $w_1$ et $\theta$) qui réalise la fonction NOT sur une entrée $x_1 \in \{0,1\}$.

**Solution :**
Pour NOT : sortie = 1 quand $x_1=0$, sortie = 0 quand $x_1=1$.

On veut : $x_1 w_1 \geq \theta$ ssi $x_1=0$.

Prenons $w_1 = -1$ et $\theta = -0.5$ (ou équivalent : biais $w_0 = 0.5$) :
- $x_1=0$ : $S = 0 \geq -0.5$ → $y=1$ ✓
- $x_1=1$ : $S = -1 < -0.5$ → $y=0$ ✓

Ou plus simplement : $w_1 = -1$, $w_0 = 0.5$, $\theta=0$ :
- $x_1=0$ : $S = 0.5 > 0$ → $y=1$ ✓
- $x_1=1$ : $S = -0.5 < 0$ → $y=0$ ✓

---

# BLOC C — GRADIENT ET DESCENTE DE GRADIENT (Exercices 19–25)

---

**Exercice 19** — Calculez le gradient de $f(x) = (x-3)^2 + 5$.

**Solution :**
$$\nabla f(x) = \frac{\partial f}{\partial x} = 2(x-3)$$

Vérification : le minimum est en $x^* = 3$ (là où $\nabla f = 0$). $f(3) = 0 + 5 = 5$. ✓

---

**Exercice 20** — Effectuez 3 itérations de la descente de gradient sur $f(x) = (x-3)^2 + 5$ avec $x_0 = 0$, $\beta = 0.2$.

**Solution :**
$\nabla f(x) = 2(x-3)$

**Itération 1** : $x_0 = 0$
$$\nabla f(0) = 2(0-3) = -6$$
$$x_1 = 0 - 0.2 \times (-6) = 0 + 1.2 = \mathbf{1.2}$$

**Itération 2** : $x_1 = 1.2$
$$\nabla f(1.2) = 2(1.2-3) = 2 \times (-1.8) = -3.6$$
$$x_2 = 1.2 - 0.2 \times (-3.6) = 1.2 + 0.72 = \mathbf{1.92}$$

**Itération 3** : $x_2 = 1.92$
$$\nabla f(1.92) = 2(1.92-3) = 2 \times (-1.08) = -2.16$$
$$x_3 = 1.92 - 0.2 \times (-2.16) = 1.92 + 0.432 = \mathbf{2.352}$$

→ Convergence progressive vers $x^* = 3$.

---

**Exercice 21** — Calculez le gradient de $g(x_1, x_2) = (x_1-2)^2 + (x_2+1)^2$.

**Solution :**
$$\nabla g = \left(\frac{\partial g}{\partial x_1}, \frac{\partial g}{\partial x_2}\right) = (2(x_1-2),\; 2(x_2+1))$$

Évaluation en $(0, 0)$ : $\nabla g(0,0) = (2(0-2), 2(0+1)) = (-4, 2)$

Le minimum global est en $(x_1^*, x_2^*) = (2, -1)$ où $\nabla g = (0,0)$. ✓

---

**Exercice 22** — Effectuez 2 itérations de la descente de gradient sur $g(x_1,x_2) = \frac{1}{2}(x_1-1)^2 + \frac{1}{2}(x_2-2)^2$ avec $(x_1^0, x_2^0) = (0, 0)$ et $\beta = 0.5$.

**Solution :**
$\nabla g = (x_1-1, x_2-2)$

**Itération 1** : $(x_1^0, x_2^0) = (0,0)$
$$\nabla g(0,0) = (0-1, 0-2) = (-1, -2)$$
$$x_1^1 = 0 - 0.5 \times (-1) = \mathbf{0.5}$$
$$x_2^1 = 0 - 0.5 \times (-2) = \mathbf{1.0}$$

**Itération 2** : $(0.5, 1.0)$
$$\nabla g(0.5, 1.0) = (0.5-1, 1.0-2) = (-0.5, -1.0)$$
$$x_1^2 = 0.5 - 0.5 \times (-0.5) = 0.5 + 0.25 = \mathbf{0.75}$$
$$x_2^2 = 1.0 - 0.5 \times (-1.0) = 1.0 + 0.5 = \mathbf{1.5}$$

→ Convergence vers $(1, 2)$. ✓

---

**Exercice 23** — Que se passe-t-il si on utilise $\beta = 2$ pour la descente de gradient sur $f(x) = (x+1)^2 - 2$, en partant de $x_0 = 0$ ? Effectuez 3 itérations.

**Solution :**
$\nabla f(x) = 2(x+1)$

**Itération 1** :
$$\nabla f(0) = 2(0+1) = 2$$
$$x_1 = 0 - 2 \times 2 = -4$$

**Itération 2** :
$$\nabla f(-4) = 2(-4+1) = -6$$
$$x_2 = -4 - 2 \times (-6) = -4 + 12 = 8$$

**Itération 3** :
$$\nabla f(8) = 2(8+1) = 18$$
$$x_3 = 8 - 2 \times 18 = 8 - 36 = -28$$

**Conclusion** : La valeur **diverge** (0 → -4 → 8 → -28...). Le taux d'apprentissage $\beta=2$ est **trop grand**, ce qui cause des oscillations qui s'amplifient. Il faut choisir un $\beta$ plus petit.

---

**Exercice 24** — Calculez le gradient de $h(w_1, w_2) = \frac{1}{2}[(1-w_1-w_2)^2 + (0-(-w_1+w_2))^2]$. C'est un exemple simplifié de MSE avec 2 exemples.

**Solution :**
Développons : $h = \frac{1}{2}[(1-w_1-w_2)^2 + (w_1-w_2)^2]$

$$\frac{\partial h}{\partial w_1} = \frac{1}{2}[2(1-w_1-w_2)(-1) + 2(w_1-w_2)(1)]$$
$$= -(1-w_1-w_2) + (w_1-w_2)$$
$$= -1+w_1+w_2+w_1-w_2 = 2w_1-1$$

$$\frac{\partial h}{\partial w_2} = \frac{1}{2}[2(1-w_1-w_2)(-1) + 2(w_1-w_2)(-1)]$$
$$= -(1-w_1-w_2) - (w_1-w_2)$$
$$= -1+w_1+w_2-w_1+w_2 = 2w_2-1$$

$$\boxed{\nabla h = (2w_1-1, 2w_2-1)}$$

Le minimum est en $w_1^*=0.5$, $w_2^*=0.5$.

---

**Exercice 25** — Expliquez pourquoi on utilise un **algorithme itératif** pour minimiser E(w) plutôt qu'une solution directe.

**Solution :**
Le vecteur solution $w^*$ **ne peut généralement pas être calculé directement** car :
1. La fonction d'erreur $E(w)$ peut être **non-convexe** (avec plusieurs minima locaux).
2. Même pour une fonction convexe, l'inversion matriciale directe peut être **trop coûteuse** en calcul pour de grandes dimensions.
3. Les réseaux de neurones profonds ont des **millions de paramètres** — résoudre analytiquement est impossible.

L'algorithme itératif de descente de gradient ajuste progressivement les poids en suivant le gradient négatif, permettant de converger vers un (bon) minimum.

---

# BLOC D — CALCUL DE LA FONCTION D'ERREUR (Exercices 26–30)

---

**Exercice 26** — Calculez la MSE pour $Y^d = (1, 0, 1, 0)$ et $\hat{Y} = (0.9, 0.1, 0.8, 0.2)$.

**Solution :**
$$\text{Loss} = \frac{1}{N}\sum_{i=1}^N (y_i^d - \hat{y}_i)^2 = \frac{1}{4}\sum$$

$$= \frac{1}{4}\left[(1-0.9)^2 + (0-0.1)^2 + (1-0.8)^2 + (0-0.2)^2\right]$$
$$= \frac{1}{4}\left[0.01 + 0.01 + 0.04 + 0.04\right]$$
$$= \frac{0.10}{4} = \boxed{0.025}$$

---

**Exercice 27** — Calculez la MSE pour $Y^d = (1, 0, 0, 0)$ et $\hat{Y} = (0.8, 0.2, 0.1, 0.7)$ (exemple du cours).

**Solution :**
$$\text{Loss} = \frac{1}{4}\left[(1-0.8)^2 + (0-0.2)^2 + (0-0.1)^2 + (0-0.7)^2\right]$$
$$= \frac{1}{4}\left[0.04 + 0.04 + 0.01 + 0.49\right]$$
$$= \frac{0.58}{4} = \boxed{0.145}$$

---

**Exercice 28** — Pour un problème de classification binaire, calculez la log-loss pour un seul exemple avec $y^d=1$ et $\hat{y}=0.8$.

**Solution :**
$$\text{Err} = -y^d \log(\hat{y}) - (1-y^d)\log(1-\hat{y})$$
$$= -1 \times \log(0.8) - (1-1) \times \log(1-0.8)$$
$$= -\log(0.8) - 0$$
$$= -(-0.2231) = \boxed{0.2231}$$

**Remarque** : Si $\hat{y}=1$ (prédiction parfaite) : $\text{Err} = -\log(1) = 0$. Si $\hat{y} \to 0$ (très mauvaise prédiction) : $\text{Err} \to +\infty$.

---

**Exercice 29** — Comparez les MSE de deux modèles :
- Modèle A : $Y^d = (1,1,0)$, $\hat{Y} = (0.9,0.8,0.1)$
- Modèle B : $Y^d = (1,1,0)$, $\hat{Y} = (0.6,0.5,0.4)$

Lequel est meilleur ?

**Solution :**
**Modèle A** :
$$\text{MSE}_A = \frac{1}{3}[(1-0.9)^2+(1-0.8)^2+(0-0.1)^2] = \frac{0.01+0.04+0.01}{3} = \frac{0.06}{3} = 0.02$$

**Modèle B** :
$$\text{MSE}_B = \frac{1}{3}[(1-0.6)^2+(1-0.5)^2+(0-0.4)^2] = \frac{0.16+0.25+0.16}{3} = \frac{0.57}{3} = 0.19$$

**Conclusion** : Le **Modèle A** est meilleur car $\text{MSE}_A = 0.02 < \text{MSE}_B = 0.19$.

---

**Exercice 30** — Pourquoi utilise-t-on la log-loss plutôt que la MSE pour la classification binaire avec sortie sigmoïde ?

**Solution :**
La **log-loss** est plus appropriée car :
1. La sortie sigmoïde représente une **probabilité** $\in ]0,1[$, et la log-loss est conçue pour mesurer l'erreur entre distributions de probabilité.
2. La log-loss pénalise **très fortement** les prédictions erronées avec haute confiance (si $y^d=1$ mais $\hat{y}\to 0$, la perte $\to +\infty$).
3. La MSE avec sigmoïde peut souffrir de **gradient très faible** dans les zones saturées, ralentissant l'apprentissage.
4. La log-loss est la **vraisemblance négative** du modèle, ce qui a une justification probabiliste solide.

---

# BLOC E — ALGORITHME DU PERCEPTRON (LOI DE HEBB) (Exercices 31–35)

---

**Exercice 31** — Reprenez l'exercice 1 du cours. Base d'apprentissage : $(1,1,1)$, $(1,-1,1)$, $(-1,1,-1)$, $(-1,-1,-1)$. Conditions : $\beta=1$, $w_1=w_2=\Theta=0$. Détaillez toutes les étapes.

**Solution :**

**Init** : $w_1=0$, $w_2=0$, $\Theta=0$

**Exemple (1)** : $x_1=1$, $x_2=1$, $y^d=1$
- $S = 1(0)+1(0)=0$; $\hat{y}=\text{signe}(0-0)=\text{signe}(0)=-1$ (convention : 0 → -1)
- Erreur : $y^d=1 \neq \hat{y}=-1$ → mise à jour :
  - $w_1 = 0 + 1 \times 1 \times 1 = 1$
  - $w_2 = 0 + 1 \times 1 \times 1 = 1$

**Exemple (2)** : $x_1=1$, $x_2=-1$, $y^d=1$, **poids actuels** : $w_1=1$, $w_2=1$
- $S = 1(1)+(-1)(1)=0$; $\hat{y}=\text{signe}(0)=-1$
- Erreur → mise à jour :
  - $w_1 = 1 + 1 \times 1 \times 1 = 2$
  - $w_2 = 1 + 1 \times (-1) \times 1 = 0$

**Exemple (3)** : $x_1=-1$, $x_2=1$, $y^d=-1$, **poids** : $w_1=2$, $w_2=0$
- $S = (-1)(2)+(1)(0)=-2$; $\hat{y}=\text{signe}(-2)=-1$
- Correct ! Pas de mise à jour.

**Exemple (4)** : $x_1=-1$, $x_2=-1$, $y^d=-1$, **poids** : $w_1=2$, $w_2=0$
- $S = (-1)(2)+(-1)(0)=-2$; $\hat{y}=\text{signe}(-2)=-1$
- Correct ! Pas de mise à jour.

**Vérification** : Tous les exemples sont-ils correctement classés avec $w_1=2$, $w_2=0$ ?
- (1): $S=2>0$ → $+1$ ✓; (2): $S=2>0$ → $+1$ ✓; (3): $S=-2<0$ → $-1$ ✓; (4): $S=-2<0$ → $-1$ ✓

**L'apprentissage est terminé. Poids finaux : $w_1=2$, $w_2=0$.**

---

**Exercice 32** — Construisez le tableau de coactivation de Hebb pour la base suivante (3 exemples) et calculez les poids finaux avec $\beta=1$, $w_1=w_2=0$ :

| $x_1$ | $x_2$ | $y^d$ |
|--------|--------|--------|
| 1 | 1 | 1 |
| -1 | 1 | -1 |
| 1 | -1 | -1 |

**Solution :**

**Init** : $w_1=0$, $w_2=0$, $\Theta=0$

**Exemple (1)** : $x_1=1$, $x_2=1$, $y^d=1$
- $S=0$; $\hat{y}=\text{signe}(0)=-1 \neq 1$
- $w_1 = 0 + 1(1)(1) = 1$; $w_2 = 0 + 1(1)(1) = 1$

**Exemple (2)** : $x_1=-1$, $x_2=1$, $y^d=-1$
- $S = (-1)(1)+(1)(1)=0$; $\hat{y}=-1=y^d$ ✓ (correct)

**Exemple (3)** : $x_1=1$, $x_2=-1$, $y^d=-1$
- $S = (1)(1)+(-1)(1)=0$; $\hat{y}=-1=y^d$ ✓ (correct)

**Retour début** : vérification exemple (1) : $S=(1)(1)+(1)(1)=2>0$ → $\hat{y}=+1$ ✓

**Tous corrects. Poids finaux : $w_1=1$, $w_2=1$.**

---

**Exercice 33** — Montrez que la loi de Hebb **échoue** sur la base de l'exercice 2 du cours (4 entrées). Expliquez pourquoi.

| $x_1$ | $x_2$ | $x_3$ | $x_4$ | $y^d$ |
|--------|--------|--------|--------|--------|
| 1 | -1 | 1 | -1 | 1 |
| 1 | 1 | 1 | 1 | 1 |
| 1 | 1 | 1 | -1 | -1 |
| 1 | -1 | -1 | 1 | -1 |

**Solution :**

**Init** : $w_1=w_2=w_3=w_4=\Theta=0$, $\beta=1$

**Exemple (1)** : $y^d=1$, $\hat{y}=\text{signe}(0)=-1$ → erreur
- $w_1=0+1(1)(1)=1$; $w_2=0+1(-1)(1)=-1$; $w_3=0+1(1)(1)=1$; $w_4=0+1(-1)(1)=-1$

**Exemple (2)** : $(1,1,1,1)$, $y^d=1$
- $S = 1(1)+1(-1)+1(1)+1(-1)=0$; $\hat{y}=-1$ → erreur
- $w_1=2$; $w_2=0$; $w_3=2$; $w_4=0$

**Exemple (3)** : $(1,1,1,-1)$, $y^d=-1$
- $S = 2+0+2+0=4>0$; $\hat{y}=+1 \neq -1$ → erreur
- $w_1=2+(-1)=1$; $w_2=0+(-1)=-1$; $w_3=2+(-1)=1$; $w_4=0+(-1)(−1)=0+1=1$

Ce processus continue sans convergence vers une solution cohérente.

**Conclusion** : Une solution existe (ex: $w=(-0.2,-0.2,0.6,0.2)$), mais la loi de Hebb ne la trouve pas. Hebb ne prend pas en compte l'erreur, il renforce la connexion selon la coactivation, ce qui ne garantit pas la convergence.

---

**Exercice 34** — Donnez un exemple de problème qui est **linéairement séparable** et un qui **ne l'est pas**. Justifiez graphiquement.

**Solution :**

**Séparable** : Porte AND
```
x₂
1 |  · (0,1)    ● (1,1)
  |
0 |  · (0,0)    · (1,0)
  +──────────────────── x₁
     0           1
```
· = classe 0, ● = classe 1. On peut tracer une ligne (ex: $x_1+x_2=1.5$) séparant (1,1) du reste. ✓

**Non séparable** : Porte XOR
```
x₂
1 |  ● (0,1)    · (1,1)
  |
0 |  · (0,0)    ● (1,0)
  +──────────────────── x₁
     0           1
```
Les ● forment un motif en diagonale : aucune droite ne peut les séparer. ✗

---

**Exercice 35** — Comment détecter qu'un algorithme de perceptron ne convergera pas ? Donnez le critère.

**Solution :**
Si, au cours de l'exécution, on rencontre **deux fois le même vecteur de poids $w$**, cela signifie que les données d'apprentissage **ne sont pas linéairement séparables** et l'algorithme ne convergera jamais.

En pratique, on fixe un **nombre maximal d'itérations** comme critère d'arrêt. Si les données sont linéairement séparables, la borne théorique d'itérations est $(N+1)^2 \cdot 2^{(N+1)\log(N+1)}$ où $N$ est le nombre d'exemples.

---

# BLOC F — RÈGLE DELTA (Exercices 36–40)

---

**Exercice 36** — Appliquez la règle delta sur l'exemple suivant : $x_1=1$, $x_2=2$, $y^d=1$, $w_1=0.3$, $w_2=-0.1$, $w_0=0.1$, $\beta=0.5$, activation = fonction signe.

**Solution :**
1. **Calcul de la sortie** :
$$S = x_1 w_1 + x_2 w_2 + w_0 = 1(0.3) + 2(-0.1) + 0.1 = 0.3 - 0.2 + 0.1 = 0.2$$
$$\hat{y} = \text{signe}(0.2) = +1$$

2. **Calcul de l'erreur** :
$$\text{Err} = y^d - \hat{y} = 1 - 1 = 0$$

3. **Mise à jour** : Err=0, donc **aucune mise à jour nécessaire**. Les poids restent inchangés.

$$\boxed{w_1=0.3, \; w_2=-0.1, \; w_0=0.1}$$

---

**Exercice 37** — Appliquez la règle delta : $x_1=1$, $x_2=0$, $y^d=1$, $w_1=0$, $w_2=0$, $w_0=0$, $\beta=0.1$, $f$ = signe.

**Solution :**
1. $S = 1(0) + 0(0) + 0 = 0$; $\hat{y} = \text{signe}(0) = -1$ (convention : 0 → -1)
2. $\text{Err} = 1 - (-1) = 2$
3. Mise à jour :
$$w_1 \leftarrow 0 + 0.1 \times 1 \times 2 = \mathbf{0.2}$$
$$w_2 \leftarrow 0 + 0.1 \times 0 \times 2 = \mathbf{0}$$
$$w_0 \leftarrow 0 + 0.1 \times 2 = \mathbf{0.2}$$

---

**Exercice 38** — Comparez la loi de Hebb et la règle delta. Donnez la formule de chacune et identifiez la différence fondamentale.

**Solution :**

| Caractéristique | Loi de Hebb | Règle delta |
|-----------------|-------------|-------------|
| Formule | $w_k \leftarrow w_k + \beta(x_k y_i^d)$ | $w_k \leftarrow w_k + \beta x_k(y_i^d - \hat{y}_i)$ |
| Utilise l'erreur | Non | Oui |
| Mise à jour si correct | Oui (si $x_k$ et $y^d$ actifs) | Non (Err=0) |
| Convergence | Pas garantie | Mieux garantie |

**Différence fondamentale** : La règle delta **tient compte explicitement de l'erreur** $\text{Err} = y^d - \hat{y}$. Si la prédiction est correcte, aucune mise à jour n'est effectuée. Hebb modifie les poids indépendamment de si la prédiction est bonne ou mauvaise.

---

**Exercice 39** — Appliquez 2 étapes de la règle delta sur la base suivante avec $w_1=0.5$, $w_2=0.5$, $w_0=-0.7$, $\beta=0.1$, $f$ = Heaviside :

| $x_1$ | $x_2$ | $y^d$ |
|--------|--------|--------|
| 1 | 1 | 1 |
| 0 | 0 | 0 |

**Solution :**

**Exemple (1)** : $x_1=1$, $x_2=1$, $y^d=1$
- $S = 1(0.5)+1(0.5)+(-0.7)=0.3$; $\hat{y}=H(0.3)=1$
- $\text{Err} = 1-1=0$ → **pas de mise à jour**

**Exemple (2)** : $x_1=0$, $x_2=0$, $y^d=0$
- $S = 0+0+(-0.7)=-0.7$; $\hat{y}=H(-0.7)=0$
- $\text{Err} = 0-0=0$ → **pas de mise à jour**

**Conclusion** : Les poids sont déjà corrects pour cette base ! Poids finaux : $w_1=0.5$, $w_2=0.5$, $w_0=-0.7$. Ce perceptron réalise déjà correctement la porte AND.

---

**Exercice 40** — Appliquez la règle delta sur cet exemple avec activation sigmoïde : $x_1=0.5$, $y^d=1$, $w_1=0$, $w_0=0$, $\beta=0.3$.

**Solution :**
1. **Calcul de la sortie** :
$$S = 0.5 \times 0 + 0 = 0$$
$$\hat{y} = \sigma(0) = \frac{1}{1+e^0} = 0.5$$

2. **Calcul de l'erreur** :
$$\text{Err} = 1 - 0.5 = 0.5$$

3. **Mise à jour** :
$$w_1 \leftarrow 0 + 0.3 \times 0.5 \times 0.5 = 0 + 0.075 = \mathbf{0.075}$$
$$w_0 \leftarrow 0 + 0.3 \times 0.5 = \mathbf{0.15}$$

---

# BLOC G — CLASSIFIEUR LINÉAIRE ET FRONTIÈRE (Exercices 41–44)

---

**Exercice 41** — Un perceptron a $w_1=1$, $w_2=-1$, $w_0=0$. Classifiez les points $(1,1)$, $(2,1)$, $(1,2)$, $(-1,0)$.

**Solution :**
Règle : si $w_1 x_1 + w_2 x_2 + w_0 = x_1 - x_2 > 0$ → classe $+1$, sinon $-1$

- $(1,1)$: $1-1=0 \leq 0$ → **classe $-1$**
- $(2,1)$: $2-1=1 > 0$ → **classe $+1$**
- $(1,2)$: $1-2=-1 \leq 0$ → **classe $-1$**
- $(-1,0)$: $-1-0=-1 \leq 0$ → **classe $-1$**

Frontière de décision : $x_1 - x_2 = 0$, c'est-à-dire la droite $x_1 = x_2$.

---

**Exercice 42** — Trouvez l'équation de la frontière de décision pour un perceptron avec $w_1=2$, $w_2=3$, $w_0=-6$ en 2D. Quels points appartiennent à quelle classe ?

**Solution :**
Frontière : $2x_1 + 3x_2 - 6 = 0$, soit $x_2 = \frac{6-2x_1}{3} = 2 - \frac{2x_1}{3}$

Classification de quelques points :
- $(0,0)$: $2(0)+3(0)-6=-6<0$ → **classe $-1$**
- $(3,0)$: $6+0-6=0$ → **sur la frontière**
- $(0,2)$: $0+6-6=0$ → **sur la frontière**
- $(3,2)$: $6+6-6=6>0$ → **classe $+1$**

La frontière est une droite passant par $(3,0)$ et $(0,2)$.

---

**Exercice 43** — Vérifiez que le perceptron AND (exemple du cours) avec $w_1=w_2=1$, $w_0=-1.5$ donne bien les sorties $f(0,0)=-1$, $f(0,1)=-1$, $f(1,0)=-1$, $f(1,1)=1$.

**Solution :**
$$f(x_1,x_2) = \text{signe}(x_1+x_2-1.5)$$

- $f(0,0) = \text{signe}(0+0-1.5) = \text{signe}(-1.5) = \mathbf{-1}$ ✓
- $f(0,1) = \text{signe}(0+1-1.5) = \text{signe}(-0.5) = \mathbf{-1}$ ✓
- $f(1,0) = \text{signe}(1+0-1.5) = \text{signe}(-0.5) = \mathbf{-1}$ ✓
- $f(1,1) = \text{signe}(1+1-1.5) = \text{signe}(0.5) = \mathbf{+1}$ ✓

Ce classifieur correspond bien à la fonction logique AND. ✓

---

**Exercice 44** — Proposez des poids pour un perceptron qui réalise la porte NOR (NON-OU) à deux entrées. La sortie est 1 uniquement si les deux entrées valent 0.

**Solution :**
Table de vérité NOR :
| $x_1$ | $x_2$ | $y$ |
|--------|--------|-----|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 0 |

On cherche $w_1, w_2, w_0$ tels que $y = 1$ ssi $w_1(0)+w_2(0)+w_0 \geq 0$ et $y=0$ pour les autres.

Prenons $w_1=-1$, $w_2=-1$, $w_0=0.5$ :
- $(0,0)$: $S = 0+0+0.5=0.5>0$ → $y=1$ ✓
- $(0,1)$: $S = 0-1+0.5=-0.5<0$ → $y=0$ ✓
- $(1,0)$: $S = -1+0+0.5=-0.5<0$ → $y=0$ ✓
- $(1,1)$: $S = -1-1+0.5=-1.5<0$ → $y=0$ ✓

**Solution : $w_1=-1$, $w_2=-1$, $w_0=0.5$**

---

# BLOC H — PROBLÈMES SYNTHÈSE ET CONCEPTUELS (Exercices 45–50)

---

**Exercice 45** — Problème complet : Un perceptron doit apprendre à classer des spams. La base d'apprentissage est :

| Exemple | $x_1$ (longueur) | $x_2$ (liens) | $y^d$ (spam=1) |
|---------|-----------------|----------------|-----------------|
| 1 | 1 | 1 | 1 |
| 2 | 0 | 0 | 0 |
| 3 | 1 | 0 | 0 |
| 4 | 0 | 1 | 1 |

Init : $w_1=0$, $w_2=0$, $w_0=0$, $\beta=1$, $f$ = Heaviside (sortie 0 ou 1).

Appliquez la règle delta sur la première epoch complète.

**Solution :**

**Convention** : $H(S)=1$ si $S>0$, $H(S)=0$ si $S\leq 0$

**Exemple (1)** : $x_1=1, x_2=1, y^d=1$
- $S=0$; $\hat{y}=0$; $\text{Err}=1-0=1$
- $w_1 \leftarrow 0+1(1)(1)=1$; $w_2 \leftarrow 0+1(1)(1)=1$; $w_0 \leftarrow 0+1(1)=1$

**Exemple (2)** : $x_1=0, x_2=0, y^d=0$; poids $(1,1,1)$
- $S=0(1)+0(1)+1=1>0$; $\hat{y}=1$; $\text{Err}=0-1=-1$
- $w_1 \leftarrow 1+1(0)(-1)=1$; $w_2 \leftarrow 1+1(0)(-1)=1$; $w_0 \leftarrow 1+1(-1)=0$

**Exemple (3)** : $x_1=1, x_2=0, y^d=0$; poids $(1,1,0)$
- $S=1(1)+0(1)+0=1>0$; $\hat{y}=1$; $\text{Err}=0-1=-1$
- $w_1 \leftarrow 1+1(1)(-1)=0$; $w_2 \leftarrow 1+1(0)(-1)=1$; $w_0 \leftarrow 0+1(-1)=-1$

**Exemple (4)** : $x_1=0, x_2=1, y^d=1$; poids $(0,1,-1)$
- $S=0(0)+1(1)+(-1)=0$; $\hat{y}=0$; $\text{Err}=1-0=1$
- $w_1 \leftarrow 0+1(0)(1)=0$; $w_2 \leftarrow 1+1(1)(1)=2$; $w_0 \leftarrow -1+1(1)=0$

**Fin de la 1ère epoch** : $w_1=0$, $w_2=2$, $w_0=0$

Vérification :
- Ex1 : $S=0+2+0=2>0$ → $\hat{y}=1=y^d$ ✓
- Ex2 : $S=0+0+0=0$ → $\hat{y}=0=y^d$ ✓
- Ex3 : $S=0+0+0=0$ → $\hat{y}=0=y^d$ ✓
- Ex4 : $S=0+2+0=2>0$ → $\hat{y}=1=y^d$ ✓

**Convergence atteinte dès la 1ère epoch. Poids finaux : $w_1=0$, $w_2=2$, $w_0=0$.**

---

**Exercice 46** — Descente de gradient complète : Effectuez 4 itérations sur $f(x) = (x+1)^2 - 2$ avec $x_0=-4$, $\beta=0.1$. Vérifiez la convergence vers $x^*=-1$.

**Solution :**
$\nabla f(x) = 2(x+1)$

| Itération $t$ | $x_t$ | $\nabla f(x_t)$ | $\Delta x = -0.1 \nabla f$ | $x_{t+1}$ |
|---------------|--------|------------------|---------------------------|------------|
| 0 | -4.000 | -6.000 | +0.600 | -3.400 |
| 1 | -3.400 | -4.800 | +0.480 | -2.920 |
| 2 | -2.920 | -3.840 | +0.384 | -2.536 |
| 3 | -2.536 | -3.072 | +0.307 | -2.229 |

La suite converge vers $x^* = -1$. À chaque étape, $|x_t - (-1)|$ diminue d'un facteur $1-2\beta = 0.8$. ✓

---

**Exercice 47** — Expliquez conceptuellement pourquoi les fonctions d'activation **non-linéaires** sont indispensables dans les réseaux de neurones.

**Solution :**
Sans fonctions d'activation non-linéaires, un réseau de neurones avec plusieurs couches serait équivalent à un réseau à **une seule couche** (car la composition de transformations linéaires reste linéaire). Formellement :

Si $f_1(x) = W_1 x + b_1$ et $f_2(x) = W_2 x + b_2$, alors :
$$f_2(f_1(x)) = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$$
C'est encore une transformation linéaire !

Les fonctions non-linéaires (ReLU, sigmoïde, tanh) permettent au réseau :
1. D'apprendre des **représentations complexes** et non-linéaires
2. De modéliser des **frontières de décision non-linéaires**
3. D'avoir la capacité d'approximer **n'importe quelle fonction** (théorème d'approximation universelle)

---

**Exercice 48** — Comparez les 3 variantes de la descente de gradient dans un tableau complet. Donnez un cas d'usage pour chacune.

**Solution :**

| Critère | Batch | Stochastique | Mini-batch |
|---------|-------|-------------|------------|
| **Calcul correction** | Après TOUS les exemples | Après 1 exemple | Après k exemples |
| **Stabilité** | Très stable | Instable/bruité | Intermédiaire |
| **Vitesse par epoch** | Lente | Rapide | Intermédiaire |
| **Convergence** | Régulière | Irrégulière/chaotique | Relativement régulière |
| **Tolérance au bruit** | Faible | Bonne | Bonne |
| **Mémoire requise** | Tout le dataset | 1 exemple | k exemples |

**Cas d'usage** :
- **Batch** : petits datasets, quand la convergence stable est prioritaire
- **Stochastique** : très grands datasets, apprentissage en ligne (streaming), quand la mémoire est limitée
- **Mini-batch** : cas général en DL (ex: batch size = 32 ou 64), meilleur compromis

---

**Exercice 49** — Un neurone avec activation sigmoïde a $w_1=1$, $w_2=-1$, $w_0=0$. Calculez la sortie et l'erreur pour $X=(1,0)$, $y^d=0.8$. Calculez les mises à jour avec $\beta=0.5$.

**Solution :**
1. **Sortie prédite** :
$$S = 1(1) + 0(-1) + 0 = 1$$
$$\hat{y} = \sigma(1) = \frac{1}{1+e^{-1}} \approx \frac{1}{1.368} \approx 0.731$$

2. **Erreur** :
$$\text{Err} = y^d - \hat{y} = 0.8 - 0.731 = 0.069$$

3. **Mise à jour des poids** :
$$w_1 \leftarrow 1 + 0.5 \times 1 \times 0.069 = 1 + 0.0345 = \mathbf{1.0345}$$
$$w_2 \leftarrow -1 + 0.5 \times 0 \times 0.069 = \mathbf{-1}$$
$$w_0 \leftarrow 0 + 0.5 \times 0.069 = \mathbf{0.0345}$$

**Remarque** : $w_2$ ne change pas car $x_2=0$ (entrée nulle → pas d'influence sur ce poids).

---

**Exercice 50** — Problème de synthèse complet. On dispose de la base :

| $x_1$ | $x_2$ | $y^d$ |
|--------|--------|--------|
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 0 | 0 | 0 |

**(a)** Ce problème est-il linéairement séparable ?
**(b)** Trouvez manuellement des poids qui résolvent ce problème.
**(c)** Calculez la MSE pour $w_1=1$, $w_2=1$, $w_0=-0.5$ (activation Heaviside).
**(d)** Quelle porte logique ce problème représente-t-il ?

**Solution :**

**(a) Séparabilité** :
En traçant les points : $(1,0)$ et $(0,1)$ sont en classe 1, $(0,0)$ est en classe 0. On peut tracer une droite $x_1+x_2=0.5$ qui sépare (0,0) des deux autres. **Oui, linéairement séparable** ✓

**(b) Recherche de poids** :
On veut $w_1(1)+w_2(0)+w_0>0$, $w_1(0)+w_2(1)+w_0>0$, $w_1(0)+w_2(0)+w_0\leq0$.

Prenons $w_1=1$, $w_2=1$, $w_0=-0.5$ :
- $(1,0)$: $1-0.5=0.5>0$ → $y=1$ ✓
- $(0,1)$: $1-0.5=0.5>0$ → $y=1$ ✓
- $(0,0)$: $0-0.5=-0.5\leq0$ → $y=0$ ✓

**(c) Calcul MSE** :
Prédictions avec Heaviside et $w_1=1$, $w_2=1$, $w_0=-0.5$ :
- $(1,0)$: $\hat{y}=1$; $(0,1)$: $\hat{y}=1$; $(0,0)$: $\hat{y}=0$

$$\text{MSE} = \frac{1}{3}[(1-1)^2+(1-1)^2+(0-0)^2] = \frac{0}{3} = \boxed{0}$$

L'erreur est nulle, ce qui confirme que les poids sont corrects !

**(d) Porte logique** :
La table de vérité $\{(1,0)→1, (0,1)→1, (0,0)→0\}$ correspond à la **porte OU** (OR) (note : $(1,1)$ n'est pas dans la base mais donnerait 1 avec ces poids).
