# Problèmes sur les fonctions d'activation

## Problème 2.1 : Évaluation de la Sigmoid 

**Donné :** Fonction sigmoid σ(x) = 1/(1 + e^(-x))

**Calculer :**
a) σ(0)
b) σ(1) 
c) σ(-1)
d) σ(5)
e) σ(-5)
f) lim(x→∞) σ(x)
g) lim(x→-∞) σ(x)

---

### Solution :

**Étape 1 : Formule de base**
```
σ(x) = 1/(1 + e^(-x))
```

**a) σ(0) :**
```
σ(0) = 1/(1 + e^0)
     = 1/(1 + 1)
     = 1/2
     = 0.5
```

**b) σ(1) :**
```
σ(1) = 1/(1 + e^(-1))
     = 1/(1 + 1/e)
     = 1/(1 + 0.368)
     = 1/1.368
     ≈ 0.731
```

**c) σ(-1) :**
```
σ(-1) = 1/(1 + e^(1))
      = 1/(1 + e)
      = 1/(1 + 2.718)
      = 1/3.718
      ≈ 0.269
```

**Méthode alternative utilisant la symétrie :**
```
σ(-x) = 1 - σ(x)
σ(-1) = 1 - σ(1)
      = 1 - 0.731
      = 0.269 ✓
```

**d) σ(5) :**
```
σ(5) = 1/(1 + e^(-5))
     = 1/(1 + 0.0067)
     ≈ 1/1.0067
     ≈ 0.993
```

**e) σ(-5) :**
```
σ(-5) = 1/(1 + e^5)
      = 1/(1 + 148.4)
      ≈ 1/149.4
      ≈ 0.0067
```

Ou en utilisant la symétrie : σ(-5) = 1 - σ(5) = 1 - 0.993 = 0.007

**f) lim(x→∞) σ(x) :**
```
Quand x → ∞ :
e^(-x) → 0
σ(x) = 1/(1 + 0) = 1
```

**g) lim(x→-∞) σ(x) :**
```
Quand x → -∞ :
e^(-x) → ∞
σ(x) = 1/(1 + ∞) = 0
```

---

### Propriétés clés :

1. **Plage :** σ(x) ∈ (0, 1) - jamais exactement 0 ou 1
2. **Point milieu :** σ(0) = 0.5
3. **Symétrie :** σ(-x) = 1 - σ(x)
4. **Monotone :** Toujours croissante
5. **Courbe en S** (sigmoïdale)

---

### Erreurs courantes :

**Dire que σ(0) = 0** - C'est 0.5, pas 0 !
**Oublier que les limites ne sont PAS atteintes** - σ s'approche de 1 et 0 mais ne les atteint jamais
**Erreurs de calcul avec e** - Utiliser e ≈ 2.718 ou une calculatrice

---

**Temps de résolution :** 8-10 minutes
**Difficulté :**
**Fréquence à l'examen :** 85%

---

## Problème 2.2 : Preuve de la dérivée de Sigmoid 

**Prouver :** Pour σ(x) = 1/(1 + e^(-x)), montrer que σ'(x) = σ(x)(1 - σ(x))

Puis calculer σ'(0), σ'(2), σ'(-2)

---

### Solution :

**Étape 1 : Dérivée en utilisant la règle de chaîne**

Soit σ(x) = 1/(1 + e^(-x)) = (1 + e^(-x))^(-1)

En utilisant la règle de chaîne :
```
σ'(x) = d/dx[(1 + e^(-x))^(-1)]
      = -1 · (1 + e^(-x))^(-2) · d/dx[1 + e^(-x)]
      = -1 · (1 + e^(-x))^(-2) · (-e^(-x))
      = e^(-x) / (1 + e^(-x))²
```

**Étape 2 : Simplifier vers la forme souhaitée**

```
σ'(x) = e^(-x) / (1 + e^(-x))²
```

Multiplier le numérateur et le dénominateur par (1 + e^(-x)) :

```
σ'(x) = [e^(-x) · (1 + e^(-x))] / [(1 + e^(-x))² · (1 + e^(-x))]
      = [e^(-x) + e^(-2x)] / (1 + e^(-x))³
```

Attendez, utilisons une approche plus astucieuse...

**Preuve alternative (plus propre) :**

```
σ'(x) = e^(-x) / (1 + e^(-x))²

Réécriture :
σ'(x) = [1 / (1 + e^(-x))] · [e^(-x) / (1 + e^(-x))]
      = σ(x) · [e^(-x) / (1 + e^(-x))]

Maintenant, que vaut e^(-x) / (1 + e^(-x)) ?

e^(-x) / (1 + e^(-x)) = (1 + e^(-x) - 1) / (1 + e^(-x))
                       = 1 - 1/(1 + e^(-x))
                       = 1 - σ(x)

Par conséquent :
σ'(x) = σ(x) · (1 - σ(x))  ✓ PROUVÉ
```

**Étape 3 : Évaluations numériques**

**σ'(0) :**
```
σ(0) = 0.5
σ'(0) = σ(0)(1 - σ(0))
      = 0.5 · (1 - 0.5)
      = 0.5 · 0.5
      = 0.25
```

**σ'(2) :**
```
σ(2) ≈ 0.881 (d'après le Problème 2.1 ou calculatrice)
σ'(2) = 0.881 · (1 - 0.881)
      = 0.881 · 0.119
      ≈ 0.105
```

**σ'(-2) :**
```
σ(-2) ≈ 0.119
σ'(-2) = 0.119 · (1 - 0.119)
       = 0.119 · 0.881
       ≈ 0.105
```

**Observation :** σ'(2) = σ'(-2) grâce à la symétrie !

---

### Points clés :

1. **Dérivée maximale en x=0 :** σ'(0) = 0.25 est la valeur la PLUS GRANDE
2. **La dérivée diminue quand |x| augmente** - Cela provoque le VANISHING GRADIENT
3. **Symétrie :** σ'(x) = σ'(-x)
4. **Plage de la dérivée :** σ'(x) ∈ (0, 0.25]

---

### Erreurs courantes :

**Oublier la règle de chaîne** - Il faut prendre en compte la fonction interne -x
**Erreurs algébriques** - La simplification est délicate
**Ne pas montrer que σ'(x) = σ(x)(1-σ(x))** - L'examen exige cette forme EXACTE

---

**Pourquoi c'est important :**
- Cette forme de dérivée est utilisée dans la backpropagation
- Montre pourquoi sigmoid cause le vanishing gradient (dérivée → 0 pour |x| grand)
- Essentiel pour comprendre l'apprentissage basé sur le gradient

**Temps de résolution :** 15-20 minutes
**Fréquence à l'examen :** 80%

---

## Problème 2.3 : Démonstration du Vanishing Gradient 
**Tâche :** Montrer numériquement pourquoi sigmoid cause le problème du "vanishing gradient"

Calculer σ'(x) pour x = 0, 1, 2, 3, 4, 5, 10

Expliquer les implications pour le deep learning.

---

### Solution :

**Étape 1 : Calculer les dérivées**

En utilisant σ'(x) = σ(x)(1 - σ(x)) :

| x  | σ(x)  | σ'(x) | Notes |
|----|-------|-------|-------|
| 0  | 0.500 | 0.250 | Maximum |
| 1  | 0.731 | 0.197 | Encore raisonnable |
| 2  | 0.881 | 0.105 | En déclin |
| 3  | 0.953 | 0.045 | Petit |
| 4  | 0.982 | 0.018 | Très petit |
| 5  | 0.993 | 0.007 | Minuscule |
| 10 | 0.9999| 0.0001| Quasi nul |

**Étape 2 : Visualisation**

```
σ'(x)
0.25 |●
     |  
0.20 |  ●
     |    
0.15 |     ●
     |        
0.10 |           ●
     |              
0.05 |                    ●
     |                         ●
0.00 |_____________________________●___
     0   1   2   3   4   5  ...  10    x
```

**Étape 3 : Implication pour le deep learning**

**Le problème :**

Dans la backpropagation, les gradients sont multipliés en arrière à travers les couches :

```
∂L/∂w₁ = ∂L/∂a₃ · σ'(z₃) · σ'(z₂) · σ'(z₁) · x
```

Si on a un réseau à 5 couches avec des activations à z = 5 :

```
∂L/∂w₁ ≈ (0.007) · (0.007) · (0.007) · (0.007) · (0.007)
        ≈ 1.7 × 10^(-12)
```

**Ce gradient est MICROSCOPIQUE !**

**Conséquences :**
1. **Les premières couches apprennent EXTRÊMEMENT lentement** - les poids changent à peine
2. **L'entraînement stagne** - le réseau semble "bloqué"
3. **Impossible d'entraîner des réseaux profonds** - plus profond = pire vanishing
4. **Limitation de représentation** - les premières couches n'apprennent pas de caractéristiques utiles

**Étape 4 : Pourquoi ReLU résout ce problème**

ReLU'(x) = {1 if x > 0, 0 if x ≤ 0}

Pour les activations positives, le gradient est TOUJOURS 1 (pas de décroissance) !

```
∂L/∂w₁ = 1 · 1 · 1 · 1 · 1 = 1  (pas de vanishing !)
```

---

### Erreurs courantes :

**Ne pas montrer les valeurs numériques** - Il faut démontrer les ordres de grandeur réels
**Confondre avec le "dying ReLU"** - Problème différent
**Ne pas expliquer l'impact de la profondeur des couches** - C'est une question de multiplication à travers les couches

---

**Question en or à l'examen :**
Cette question teste :
- Le calcul numérique (Problème 2.2)
- La compréhension conceptuelle (pourquoi c'est un problème)
- La connaissance des solutions (ReLU)

**Temps de résolution :** 15-18 minutes
**Difficulté :** 
**Fréquence à l'examen :** 60% (mais vaut beaucoup de points)

---

## Problème 2.4 : Fonction Tanh 

**Donné :** tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

**Tâches :**
a) Calculer tanh(0), tanh(1), tanh(-1)
b) Prouver : tanh'(x) = 1 - tanh²(x)
c) Comparer avec sigmoid

---

### Solution :

**Étape 1 : Évaluations**

**a) tanh(0) :**
```
tanh(0) = (e^0 - e^0)/(e^0 + e^0)
        = (1 - 1)/(1 + 1)
        = 0/2
        = 0
```

**b) tanh(1) :**
```
tanh(1) = (e - e^(-1))/(e + e^(-1))
        = (2.718 - 0.368)/(2.718 + 0.368)
        = 2.350/3.086
        ≈ 0.762
```

**c) tanh(-1) :**
```
tanh(-1) = (e^(-1) - e^1)/(e^(-1) + e^1)
         = (0.368 - 2.718)/(0.368 + 2.718)
         = -2.350/3.086
         ≈ -0.762
```

Note : tanh(-x) = -tanh(x) (fonction impaire)

---

**Étape 2 : Preuve de la dérivée**

En partant de tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

En utilisant la règle du quotient : (u/v)' = (u'v - uv')/v²

Soit :
- u = e^x - e^(-x), donc u' = e^x + e^(-x)
- v = e^x + e^(-x), donc v' = e^x - e^(-x)

```
tanh'(x) = [(e^x + e^(-x))(e^x + e^(-x)) - (e^x - e^(-x))(e^x - e^(-x))] / (e^x + e^(-x))²
         = [(e^x + e^(-x))² - (e^x - e^(-x))²] / (e^x + e^(-x))²
```

Développement :
```
(e^x + e^(-x))² = e^(2x) + 2 + e^(-2x)
(e^x - e^(-x))² = e^(2x) - 2 + e^(-2x)

Différence :
(e^x + e^(-x))² - (e^x - e^(-x))² = (e^(2x) + 2 + e^(-2x)) - (e^(2x) - 2 + e^(-2x))
                                   = 4
```

Par conséquent :
```
tanh'(x) = 4 / (e^x + e^(-x))²
```

Maintenant, exprimer en fonction de tanh(x) :
```
tanh²(x) = [(e^x - e^(-x))/(e^x + e^(-x))]²
         = (e^x - e^(-x))² / (e^x + e^(-x))²

1 - tanh²(x) = 1 - (e^x - e^(-x))² / (e^x + e^(-x))²
              = [(e^x + e^(-x))² - (e^x - e^(-x))²] / (e^x + e^(-x))²
              = 4 / (e^x + e^(-x))²
              = tanh'(x)  ✓ PROUVÉ
```

---

**Étape 3 : Comparaison avec sigmoid**

| Propriété | Sigmoid σ(x) | Tanh |
|----------|--------------|------|
| Plage | (0, 1) | (-1, 1) |
| Point milieu | 0.5 | 0 |
| Centré sur zéro |  Non |  Oui |
| Symétrie | σ(-x) = 1 - σ(x) | tanh(-x) = -tanh(x) |
| Dérivée | σ'(x) = σ(x)(1-σ(x)) | tanh'(x) = 1 - tanh²(x) |
| Dérivée max | 0.25 en x=0 | 1.0 en x=0 |
| Vanishing gradient |  Oui |  Oui (mais moins sévère) |

**Avantage clé de tanh :** 
- Les sorties centrées sur zéro signifient que les activations peuvent être positives OU négatives
- Cela aide au flux de gradient dans les réseaux profonds
- Souffre toujours du vanishing gradient pour |x| grand

---

### Erreurs courantes :

 **Confondre la plage avec sigmoid** - tanh est (-1,1), pas (0,1)
 **Ne pas montrer que tanh'(x) = 1 - tanh²(x)** - Il faut dériver cette forme
 **Oublier que tanh est une fonction impaire** - tanh(-x) = -tanh(x)

---

**Temps de résolution :** 18-22 minutes
**Difficulté :**
**Fréquence à l'examen :** 65%

---

## Problème 2.5 : Fonction ReLU 

**Donné :** ReLU(x) = max(0, x) = {x if x > 0, 0 if x ≤ 0}

**Tâches :**
a) Évaluer ReLU(-2), ReLU(0), ReLU(3)
b) Calculer ReLU'(x)
c) Expliquer pourquoi ReLU n'a pas de vanishing gradient
d) Qu'est-ce que le problème du "dying ReLU" ?

---

### Solution :

**Étape 1 : Évaluations**

**a) ReLU(-2) :**
```
ReLU(-2) = max(0, -2) = 0
```

**b) ReLU(0) :**
```
ReLU(0) = max(0, 0) = 0
```

**c) ReLU(3) :**
```
ReLU(3) = max(0, 3) = 3
```

Simple : Sortie = entrée si positive, sinon 0.

---

**Étape 2 : Dérivée**

```
ReLU'(x) = {1  if x > 0
           {0  if x < 0
           {indéfinie en x = 0 (mais on utilise 0 ou 1 en pratique)
```

**Pourquoi ?**

Pour x > 0 : ReLU(x) = x, donc d/dx[x] = 1
Pour x < 0 : ReLU(x) = 0, donc d/dx[0] = 0
En x = 0 : La dérivée est techniquement indéfinie (coin anguleux)

**En pratique :** On pose ReLU'(0) = 0 ou 1 (peu d'importance car c'est un seul point)

---

**Étape 3 : Pas de vanishing gradient**

**Point clé :** Pour tout x > 0, ReLU'(x) = 1

Dans un réseau profond :
```
∂L/∂w₁ = ... · ReLU'(z₃) · ReLU'(z₂) · ReLU'(z₁) · ...
       = ... · 1 · 1 · 1 · ...
       = ... (pas de décroissance multiplicative !)
```

**Comparaison avec sigmoid :**
```
∂L/∂w₁ = ... · σ'(z₃) · σ'(z₂) · σ'(z₁) · ...
       = ... · 0.1 · 0.05 · 0.2 · ...
       = ... · 0.001 (décroissance massive !)
```

**ReLU résout le vanishing gradient pour les activations positives.**

---

**Étape 4 : Problème du dying ReLU**

**Le problème :**

Si la somme pondérée d'un neurone devient négative, ReLU renvoie 0.
Si le gradient rétropropagé est 0, les poids ne se mettent JAMAIS à jour.
Le neurone devient "mort" - il renvoie toujours 0.

**Exemple :**

Supposons qu'un neurone a un grand bias négatif : w·x + b = -100
- ReLU(-100) = 0
- ReLU'(-100) = 0
- Gradient = 0
- Mise à jour des poids : Δw = 0 (pas d'apprentissage !)
- Le neurone est bloqué à renvoyer 0 pour toujours

**Causes :**
1. Mauvaise initialisation des poids (grandes valeurs négatives)
2. Taux d'apprentissage trop élevé (les poids sautent vers la région négative)
3. Beaucoup d'exemples d'entraînement négatifs poussant les poids vers le bas

**Solutions :**
1. Utiliser Leaky ReLU : max(0.01x, x) - petit gradient même pour x < 0
2. Utiliser PReLU (paramétrique) : max(αx, x) où α est appris
3. Initialisation soignée (par ex., initialisation de He)
4. Réduire le taux d'apprentissage

---

**Étape 5 : Tableau récapitulatif**

| Propriété | ReLU | Sigmoid | Tanh |
|----------|------|---------|------|
| Plage | [0, ∞) | (0, 1) | (-1, 1) |
| Dérivée pour x grand | 1 | ~0 | ~0 |
| Vanishing gradient |  Non |  Oui |  Oui |
| Problème de neurone mort |  Oui |  Non |  Non |
| Coût de calcul | Très faible | Moyen | Moyen |
| Centré sur zéro |  Non |  Non |  Oui |

**Pourquoi ReLU domine le deep learning moderne :**
- Rapide à calculer
- Pas de vanishing gradient (pour x > 0)
- Fonctionne empiriquement très bien
- Malgré le risque de dying ReLU, les avantages l'emportent sur les inconvénients

---

### Erreurs courantes :
**Dire que ReLU'(2) = 2** - La dérivée est 1, pas x !
**Confondre dying ReLU avec vanishing gradient** - Problèmes différents
**Ne pas expliquer pourquoi ReLU aide** - Il faut mentionner le gradient constant = 1

---

**Conseil d'examen :** 
Si on vous demande "Pourquoi ReLU est meilleur que sigmoid ?", répondez :
1. Pas de vanishing gradient (gradient = 1 pour x > 0)
2. Calcul plus rapide (juste une opération max)
3. Fonctionne empiriquement mieux dans les réseaux profonds

**Temps de résolution :** 12-15 minutes
**Fréquence à l'examen :** 90% (ReLU est omniprésent dans le deep learning moderne)

---

## Résumé & Prochaines étapes

**Vous maîtrisez maintenant :**
- Évaluation et dérivée de sigmoid ✓
- Problème du vanishing gradient ✓
- Propriétés de tanh et comparaison ✓
- Avantages de ReLU et dying ReLU ✓

**Stratégie de pratique :**
1. Faire tous les calculs sans calculatrice d'abord (développer l'intuition)
2. Puis vérifier avec une calculatrice
3. Être capable de dessiner les trois fonctions de mémoire
4. Connaître les formes des dérivées PAR CŒUR : σ'(x) = σ(1-σ), tanh' = 1 - tanh², ReLU' = {0,1}
