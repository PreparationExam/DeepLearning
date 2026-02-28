# RÉSUMÉ COMPLET DÉTAILLÉ — Chapitre 1 : Concepts Fondamentaux d'un Réseau de Neurones

---

## 1.1 Machine Learning et Deep Learning

### 1.1.1 Machine Learning (ML)

Le **machine learning** est un ensemble de méthodes permettant de **détecter automatiquement des régularités dans des données**. C'est un domaine de l'intelligence artificielle (IA).

Chaque méthode ML est associée à un algorithme d'optimisation spécifique :
- Descente de gradient → modèles linéaires
- Algorithme CART → arbres de décision
- Marge maximum → SVM (Support Vector Machines)

Les régularités détectées servent à **décrire** ou **prédire** des caractéristiques des données.

Le ML se présente sous deux formes principales :
1. **Apprentissage supervisé**
2. **Apprentissage non supervisé**

---

### 1.1.1.1 Apprentissage Supervisé

L'apprentissage supervisé correspond aux situations où les **données sont étiquetées** : pour chaque exemple d'entrée, on dispose de la réponse correcte.

**Base d'entraînement** (N exemples étiquetés) :

$$B = (X_i, y_i^d)_{1 \leq i \leq N}$$

- $X_i$ : vecteur d'entrée de dimension $n$, associé à l'exemple $i$
- $(x_{i1}, \ldots, x_{in})$ : attributs ou **caractéristiques (features)** décrivant l'exemple
- $n$ : nombre total de caractéristiques
- $y_i^d$ : sortie désirée (étiquette)

**Deux types de problèmes supervisés :**

| Type | Définition | Sortie |
|------|-----------|--------|
| **Classification** | Associer chaque entrée à une catégorie | $y_i^d \in \{1, \ldots, C\}$ (entier) |
| **Régression** | Prédire une valeur numérique continue | $y_i^d \in \mathbb{R}$ (réel) |

**Exemples de régression :**
- Météorologie : X = altitude d'une station → y = température
- Santé : X = âge → y = tension artérielle

**Exemples de classification :**
- Classification d'images : X = pixels → y = espèce animale
- Analyse de sentiments : X = phrase → y = sentiment (joie, peur, …)
- Reconnaissance vocale : X = échantillon de voix → y = identité

**Objectif :** trouver $f$ telle que $f(X) \approx y$, apprise à partir de $(X_i, y_i^d)_{1 \leq i \leq N}$.

---

### 1.1.1.2 Apprentissage Non Supervisé

Les données sont **non étiquetées** : on ne dispose que des entrées.

$$B = (X_i)_{1 \leq i \leq N}$$

**Objectif :** découvrir automatiquement des structures cachées ou des régularités dans les données.

**Tâche principale : le Clustering**
- Regrouper les données en sous-ensembles appelés **clusters**
- Les données d'un même cluster se ressemblent
- Les clusters représentent des comportements ou profils distincts

**Autres tâches :**
- **Réduction de dimension** : simplifier les données tout en conservant l'essentiel (ex. ACP/PCA)
- **Détection d'anomalies** : trouver des données très différentes des autres

**Difficultés :**
- On ne sait pas à l'avance quelles régularités sont intéressantes
- Il n'existe pas de réponse correcte pour vérifier si le résultat est bon
- La qualité dépend fortement de la méthode choisie

> ⚠️ Dans ce cours, on se concentre **exclusivement sur l'apprentissage supervisé**.

---

### 1.1.2 Bases du Machine Learning

ML = trouver automatiquement le **meilleur modèle** à partir des données.

Un modèle ML = une **fonction mathématique** $f(x)$ : entrée → prédiction.

**Processus général d'apprentissage :**

1. **Initialisation** : les paramètres du modèle (poids, coefficients) sont initialisés **aléatoirement**
2. **Prédiction** : le modèle calcule $f(x)$
3. **Évaluation** : on calcule la **fonction coût** (mesure de l'erreur)
4. **Amélioration** : un algorithme d'optimisation ajuste les paramètres
5. **Répétition** jusqu'à obtenir une performance satisfaisante
6. **Déploiement** : le modèle final est utilisé en conditions réelles

**La fonction coût :**
- Mesure l'erreur entre prédictions $f(x)$ et valeurs réelles $y$
- Petite valeur → bonnes prédictions ; grande valeur → mauvaises prédictions
- Joue le rôle de **"boussole"** pour l'apprentissage

**La descente de gradient :**
- Algorithme d'optimisation le plus répandu en ML
- Processus itératif : calcule la direction où la fonction coût diminue le plus vite, puis met à jour les paramètres dans cette direction

---

### 1.1.3 Bases du Deep Learning

Le **Deep Learning (DL)** est une **sous-catégorie du ML**.

| ML classique | Deep Learning |
|-------------|---------------|
| Modèle unique | Réseau de nombreuses fonctions interconnectées |
| Problèmes simples/modérés | Problèmes très complexes |

**Origine historique :**
- **1943** : Warren McCulloch et Walter Pitts proposent le **neurone formel** (premier modèle mathématique du neurone biologique)
- Ce modèle a constitué la base des premiers réseaux de neurones artificiels
- Capacité à résoudre des opérations logiques simples (ET, OU, NON)

**Hiérarchie :** IA ⊃ Machine Learning ⊃ Deep Learning

---

## 1.2 Du Biologique à l'Artificiel

### 1.2.1 Neurones Biologiques

Un neurone biologique = cellule spécialisée du système nerveux pour **transmettre et traiter des signaux électriques**.

**Composants clés :**

| Composant | Rôle |
|-----------|------|
| **Dendrites** | Extensions qui **reçoivent** les signaux d'autres neurones → entrées |
| **Synapses** | Points de contact entre neurones → transmission des signaux |
| **Axone** | Prolongement qui **conduit** le signal → sortie du neurone |
| **Noyau** | Rôle central dans le fonctionnement cellulaire et l'activation |

**Fonctionnement :**
- Un neurone possède environ **10 000 synapses**
- Lorsque la somme des signaux dépasse un certain **seuil** → le neurone s'active → génère un **potentiel d'action**
- Ce signal circule le long de l'axone jusqu'aux synapses → communication avec d'autres neurones

---

### 1.2.2 Neurone Artificiel de McCulloch et Pitts (1943)

Premier modèle mathématique simplifié du neurone biologique : le **neurone formel**.

**Entrées :** vecteur binaire $X = [x_1, x_2, \ldots, x_n]^T$ (chaque $x_i \in \{0, 1\}$)

**Sortie :** binaire $y \in \{0, 1\}$

**Poids synaptique $w_i$** : représente l'importance de la connexion $x_i$.

**Deux phases de fonctionnement :**

**Phase 1 — Agrégation :**
$$S = \sum_{i=1}^{n} x_i w_i$$

**Phase 2 — Activation :**
$$y = f\left(\sum_{i=1}^{n} x_i w_i\right) = \begin{cases} 1 & \text{si } \sum_{i=1}^{n} x_i w_i \geq \theta \\ 0 & \text{sinon} \end{cases}$$

La sortie $y$ est également appelée **activation** du neurone.

> Aujourd'hui, les réseaux utilisent des fonctions d'activation **continues** et traitent des données **non binaires**.

---

### 1.2.3 Fonctions d'Activation

La **fonction d'activation** (aussi appelée **fonction de transfert**) détermine la réponse d'un neurone en fonction de sa stimulation. Elle agit comme un seuil.

Pour une entrée réelle $x \in \mathbb{R}$, elle produit $y = f(x)$.

#### 1. Fonction Heaviside (marche d'escalier)

Fonction originale du neurone formel de McCulloch et Pitts.

$$y = H(x) = \begin{cases} 0 & \text{si } x < 0 \\ 1 & \text{si } x \geq 0 \end{cases}$$

**Exemple concret :** diode électronique qui s'allume ($y=1$) si l'intensité dépasse zéro ($x > 0$).

**Schéma ASCII :**
```
y
1 |--------
  |
0 |___
      0    x
```

#### 2. Fonction Sigmoïde

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Sortie dans $]0, 1[$
- Ressemble à Heaviside mais avec **transition douce et continue** (courbe en "S")

**Avantages :**
- Continue et dérivable → facilite l'optimisation et la descente de gradient
- Activation progressive autour de 0 → réponse graduelle du neurone

#### 3. Fonction Tangente Hyperbolique (tanh)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- Sortie dans $]-1, +1[$
- Fonction **impaire** (symétrique par rapport à l'origine) → aide à **centrer les données**
- Similaire à la sigmoïde mais centrée en 0

#### 4. Fonction ReLU (Rectified Linear Unit)

$$f(x) = \max(0, x) = \begin{cases} 0 & \text{si } x < 0 \\ x & \text{si } x \geq 0 \end{cases}$$

**La fonction d'activation la plus utilisée en Deep Learning moderne.**

**Avantages :**
- Sortie nulle pour les entrées négatives, proportionnelle pour les positives
- Facilite l'apprentissage
- **Limite le problème de disparition du gradient** rencontré avec la sigmoïde

> ⚠️ Le choix de la fonction d'activation a un impact direct sur l'efficacité et la convergence du réseau.

---

### 1.2.4 Réseaux de Neurones Réalisant des Calculs Logiques

Avec McCulloch-Pitts, poids fixés à 1, on obtient différentes opérations logiques selon $\theta$ :

| Seuil $\theta$ | Opération | Sortie |
|---------------|-----------|--------|
| $\theta = 1$ | **OU** | 1 si au moins une entrée vaut 1 |
| $\theta = 2$ | **ET** | 1 uniquement si les deux entrées valent 1 |

**Limitation fondamentale :** les poids synaptiques doivent être **définis manuellement** → pas d'apprentissage automatique.

**Solution : le Perceptron (1957)** — Frank Rosenblatt invente le premier algorithme permettant d'**apprendre automatiquement** les poids à partir d'exemples.

---

## 1.3 Le Perceptron

### 1.3.1 Définition

Le **perceptron** = modèle le plus ancien et simple des réseaux de neurones artificiels.

- Introduit par **Frank Rosenblatt en 1957**
- Algorithme d'apprentissage **supervisé**
- Utilisé pour la **classification binaire**

Un perceptron reçoit plusieurs entrées, leur applique des poids, calcule une somme pondérée.

**Biais :** une entrée supplémentaire $x_0 = 1$, associée au poids $w_0 = -\Theta$ (incorpore le seuil $\Theta$ dans le calcul).

---

### 1.3.2 Fonctionnement

**Calcul de la somme pondérée :**
$$S = \sum_{i=1}^{n} x_i w_i$$

**Application de la fonction d'activation seuil :**
$$\text{sortie} = \begin{cases} 1 & \text{si } \sum_{i=1}^{n} x_i w_i > \Theta \\ 0 & \text{sinon} \end{cases}$$

Le perceptron sépare les données en **deux classes distinctes**. Si les données sont **linéairement séparables**, il apprend une frontière de décision correcte.

---

### 1.3.3 Rôle des Poids

| Poids $w_i$ | Effet sur l'entrée $x_i$ |
|-------------|--------------------------|
| Positif | Renforce l'impact de l'entrée |
| Négatif | Diminue l'impact |
| Proche de 0 | Entrée peu pertinente |

Le **biais** $w_0$ permet de **déplacer la frontière de décision** → essentiel pour modéliser correctement les données.

---

### 1.3.4 Apprentissage du Perceptron

Si le perceptron fait une erreur, il met à jour ses poids :

$$w_i^{(t+1)} = w_i^{(t)} + \eta (y - \hat{y}) x_i$$

où :
- $y$ = vraie étiquette
- $\hat{y}$ = prédiction du perceptron
- $\eta$ = taux d'apprentissage (paramètre positif)

---

### 1.3.5 Applications et Limites

**Applications historiques :**
- Détection de formes
- Reconnaissance de caractères manuscrits
- Filtrage de spam
- Classification d'images simples

**Limite fondamentale :** ne peut apprendre que des problèmes **linéairement séparables**.

Les tâches plus complexes nécessitent des **MLP (perceptrons multicouches)** ou des **réseaux profonds**.

---

### 1.3.6 Implémentation Python

**Porte AND** (w = [0.5, 0.5], w0 = -0.7) :
```python
import numpy as np
def and_perceptron(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    w0 = -0.7
    S = np.sum(w * x) + w0
    return 0 if S <= 0 else 1
```

**Porte NAND** (w = [-0.5, -0.5], w0 = 0.7) :
```python
def nand_perceptron(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    w0 = 0.7
    S = np.sum(w * x) + w0
    return 0 if S <= 0 else 1
```

---

### 1.3.7 Perceptron : Classifieur Linéaire

**Fonction de décision :**
$$\text{signe}(x_1, \ldots, x_n) = \begin{cases} 1 & \text{si } \sum_{i=1}^{n} w_i x_i + w_0 > 0 \\ -1 & \text{sinon} \end{cases}$$

**Frontière de décision** = hyperplan :
$$\sum_{i=1}^{n} w_i x_i + w_0 = 0$$

**Exemple :** poids $(w_1, w_2) = (1, 1)$, biais $w_0 = -1.5$

Frontière : $x_1 + x_2 - 1.5 = 0$

| $(x_1, x_2)$ | $x_1 + x_2 - 1.5$ | Sortie |
|---|---|---|
| (0,0) | -1.5 | -1 |
| (0,1) | -0.5 | -1 |
| (1,0) | -0.5 | -1 |
| (1,1) | +0.5 | **+1** |

→ Correspond à la fonction **AND**.

---

## 1.4 Algorithme d'Apprentissage du Perceptron

### 1.4.1 Loi de Hebb

**Principe :** Rosenblatt s'est inspiré du neuropsychologue **Donald Hebb**.

**Loi de Hebb :**
- Lorsque deux neurones sont **activés simultanément** → connexion **renforcée**
- Activation asynchrone → connexion **affaiblie ou éliminée**

**Tableau de coactivation :**

| $x_i$ | $x_j$ | $\partial w(i,j) = x_i \cdot x_j$ |
|--------|--------|-----------------------------------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| **1** | **1** | **+** |

**Règle de mise à jour :**
$$w_i = w_i + \beta(x_i x_j) \quad \Leftrightarrow \quad \Delta w(i,j) = \beta(x_i x_j)$$

$\beta$ = **taux d'apprentissage** (intensité de la modification).

**Algorithme (déroulement général) :**
1. Initialiser aléatoirement les poids $w_i$ et le seuil $\Theta$ (valeurs faibles)
2. Sélectionner un exemple $(X_i, y_i^d)$ dans la base $B$
3. Calculer la sortie : $\hat{y}_i = \text{signe}(S - \Theta)$ où $S = \sum_k x_k w_k$
   - Si $S - \Theta > 0 \Rightarrow \hat{y}_i = +1$, sinon $\hat{y}_i = -1$
4. Si sortie incorrecte ($y_i^d \neq \hat{y}_i$) : mettre à jour $w_k = w_k + \beta(x_k y_i^d)$
5. Répéter tant que tous les exemples de $B$ ne sont pas classés correctement
6. Retourner les poids $w_i$ obtenus

**Limitation :** La règle de Hebb **ne converge pas toujours**, même lorsque le problème est linéairement séparable.

---

### 1.4.2 Règle Delta

#### 1.4.2.1 Principe

Objectif : ajuster les poids pour que $\hat{y}$ soit le plus proche possible de $y^d$.

Le perceptron compare la sortie désirée avec la sortie prédite, puis modifie les poids dans le sens qui **réduit la différence**.

#### 1.4.2.2 Modèle du Perceptron

$$\hat{y}_i = f\left(\sum_{i=1}^{n} x_i w_i + w_0\right)$$

où $f$ = fonction d'activation (fonction signe ou sigmoïde).

#### 1.4.2.3 Mise à Jour des Poids

**Principe clé :** si l'erreur est grande → modification des poids importante.

$$w_i = w_i + \beta \cdot x_i \cdot (y_i^d - \hat{y}_i)$$

Forme compacte :
$$w_i = w_i + \beta \cdot x_i \cdot \text{Err}_i$$

où $\text{Err}_i = y_i^d - \hat{y}_i$

**Comportement :**
- $\text{Err}_i = 0$ → poids inchangé
- Prédiction trop faible → poids augmente
- Prédiction trop forte → poids diminue

#### 1.4.2.4 Processus d'Apprentissage (Étapes)

Pour chaque exemple d'apprentissage :

1. **Calcul de la sortie prédite :** $\hat{y} = f(w \cdot x + w_0)$
2. **Calcul de l'erreur :** $\text{Err} = y^d - \hat{y}$
3. **Mise à jour des poids :** $w_i \leftarrow w_i + \beta x_i (\text{Err})$
4. **Mise à jour du biais :** $w_0 \leftarrow w_0 + \beta (\text{Err})$
5. **Répétition** pour tous les exemples, sur plusieurs **époques**, jusqu'à ce que les erreurs soient faibles ou nulles.

#### 1.4.2.5 Fonction d'Erreur (Risque Empirique)

$$\text{loss} = E(w) = \frac{1}{N} \sum_{i=1}^{N} \text{Err}_i(w)$$

Cette quantité mesure la qualité du modèle. **La fonction d'erreur dépend directement des poids.**

**Choix selon le type de problème :**

| Problème | Fonction d'erreur | Nom |
|----------|-------------------|-----|
| Régression | $\text{Err}_i(w) = (y_i^d - \hat{y}_i)^2$ | MSE (Mean Squared Error) |
| Classification binaire | $\text{Err}_i(w) = -y_i^d \log(\hat{y}_i) - (1-y_i^d)\log(1-\hat{y}_i)$ | Log-loss |
| Classification multiclasse | $\text{Err}_i(w) = -\sum_{k=1}^{K} y_k^d \log(\hat{y}_k)$ | Entropie croisée |

**Exemple illustratif :**
$$Y^d = \begin{pmatrix}1\\0\\0\\0\end{pmatrix}, \quad \hat{Y} = \begin{pmatrix}0.8\\0.2\\0.1\\0.7\end{pmatrix}$$

$$\text{Loss (MSE)} = \frac{1}{4}\left[(1-0.8)^2 + (0-0.2)^2 + (0-0.1)^2 + (0-0.7)^2\right] = 0.145$$

#### 1.4.2.6 Minimisation d'Erreur

Apprentissage = problème d'**optimisation** : trouver $w^*$ qui minimise $E(w)$.

- **Fonction convexe** → minimum unique → convergence garantie
- **Fonction non-convexe** → plusieurs minima locaux → résultat dépend du point de départ

Le vecteur solution $w^*$ ne peut généralement pas être calculé directement → **algorithme itératif**.

---

#### 1.4.2.7 Gradient d'une Fonction

**Définition :** le gradient d'une fonction $f$ de $n$ variables est le vecteur de ses dérivées partielles :

$$\nabla f(x_1, x_2, \ldots, x_n) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

**Propriété importante :** le gradient indique la direction de **variation la plus rapide** (augmentation). Pour **minimiser**, on se déplace dans la direction **opposée** au gradient.

**Exemple 1 :** $f(x) = (x+1)^2 - 2$
$$\nabla f(x) = \frac{\partial f}{\partial x} = 2(x+1)$$
- $\nabla f(-2) = -2$, $\nabla f(0) = 2$

**Exemple 2 :** $g(x_1, x_2) = \frac{1}{2}(x_1-1)^2 + \frac{1}{2}(x_2-2)^2$
$$\nabla g(x_1, x_2) = (x_1 - 1, \; x_2 - 2)$$
- $\nabla g(0,0) = (-1,-2)$, $\nabla g(1,3) = (0,1)$

---

#### 1.4.2.8 Algorithme de Descente du Gradient

Pour trouver le minimum de $f(x)$ :

1. Choisir une valeur initiale $x_0$ et un taux d'apprentissage $\beta > 0$
2. Répéter jusqu'à convergence :
   - (a) Calculer la correction : $\Delta x = -\beta \nabla f(x_{t-1})$
   - (b) Mettre à jour : $x_t = x_{t-1} + \Delta x$

**Convergence :** $x_t$ converge vers le minimum quand il ne change plus.

**Exemple :** $f(x) = (x+1)^2 - 2$, $\nabla f(x) = 2(x+1)$, $x_0 = -4$, $\beta = 0.1$

| Itération | $\nabla f(x_t)$ | $x_{t+1}$ |
|-----------|----------------|-----------|
| 0 | $2(-4+1) = -6$ | $-4 + 0.1 \times 6 = -3.4$ |
| 1 | $2(-3.4+1) = -4.8$ | $-3.4 + 0.1 \times 4.8 = -2.92$ |
| … | … | … |
| ∞ | 0 | **$x^* = -1$** |

**Notion d'epoch :** une **epoch** = une itération complète sur l'ensemble des exemples d'apprentissage.

---

#### 1.4.2.9 Variantes de la Descente de Gradient

| Variante | Description | Avantages | Inconvénients |
|---------|-------------|-----------|----------------|
| **Batch (hors-ligne)** | Correction après l'ensemble des exemples | Très stable | Lent sur grands datasets |
| **Stochastique (en-ligne)** | Correction pour un seul exemple tiré au hasard | Rapide, tolère le bruit | Convergence irrégulière (chaotique) |
| **Mini-batch** | Correction sur un petit sous-ensemble | Compromis batch/stochastique | — |

---

#### 1.4.2.9 Algorithme d'Apprentissage Complet

**Mises à jour des poids et du biais :**

$$\Delta w_i = -\beta \nabla E(w) = -\beta \frac{\partial E}{\partial w_i} = \frac{\beta}{N} X_i^T (y_i^d - \hat{y}_i)$$

$$\Delta w_0 = -\beta \nabla E(w_0) = \frac{\beta}{N} \sum_{i=0}^{n} (y_i^d - \hat{y}_i)$$

**Déroulement :**
1. Initialiser aléatoirement $w_i$ et $w_0$
2. Tant qu'il y a des mises à jour :
   - Pour chaque $i \in [0, n]$ (batch) ou quelques $i$ (mini-batch) :
     - Calculer $\hat{y}_i = \text{fct\_act}(S + w_0)$
     - Si $y_i^d \neq \hat{y}_i$ : mettre à jour $w_i$ et $w_0$
3. Retourner $w_i$

---

#### 1.4.2.10 Convergence de l'Algorithme

**Détection de non-séparabilité :** si on rencontre **deux fois le même vecteur $w$** → les données ne sont **pas linéairement séparables**.

**Borne maximale d'itérations** (données linéairement séparables) :
$$(N+1)^2 \cdot 2^{(N+1)\log(N+1)}$$

**Remarques importantes :**
- $\beta$ trop grand → risque d'**oscillations** autour du minimum
- $\beta$ trop petit → nombre **élevé d'itérations** nécessaires
- En pratique, $\beta$ est souvent **diminué progressivement**
- L'algorithme ne converge que si les deux classes sont **bien séparées**
- Le perceptron est **difficilement généralisable** à plus de deux classes
