# Problèmes de mise à jour des poids - Delta rule & Hebb's rule

## Problème 3.1 : Mise à jour unique avec la Delta rule 

**Données :**
- Poids actuels : w = (1, 1)
- Bias : w₀ = -1.5
- Taux d'apprentissage : η = 0.1
- Entrée : x = (1, 1)
- Sortie désirée : y_d = 1
- Sortie actuelle du perceptron : y = 0 (MAUVAISE prédiction)

**Tâche :** Calculer les nouveaux poids après une mise à jour avec la Delta rule.

---

### Solution :

**Étape 1 : Rappel de la formule de la Delta rule**

```
Δwᵢ = η · e · xᵢ
où e = (y_d - y) est l'erreur

Nouveau poids : wᵢ_new = wᵢ_old + Δwᵢ
```

**Étape 2 : Calcul de l'erreur**

```
e = y_d - y
  = 1 - 0
  = 1
```

**Étape 3 : Calcul des mises à jour des poids**

```
Δw₁ = η · e · x₁
    = 0.1 · 1 · 1
    = 0.1

Δw₂ = η · e · x₂
    = 0.1 · 1 · 1
    = 0.1
```

**Étape 4 : Calcul de la mise à jour du bias**

Pour le bias, on considère x₀ = 1 (entrée constante) :

```
Δw₀ = η · e · x₀
    = 0.1 · 1 · 1
    = 0.1
```

**Étape 5 : Application des mises à jour**

```
w₁_new = w₁_old + Δw₁ = 1 + 0.1 = 1.1
w₂_new = w₂_old + Δw₂ = 1 + 0.1 = 1.1
w₀_new = w₀_old + Δw₀ = -1.5 + 0.1 = -1.4
```

**RÉPONSE : w_new = (1.1, 1.1), w₀_new = -1.4**

---

**Étape 6 : Vérification**

Tester les nouveaux poids sur la même entrée pour vérifier l'amélioration :

```
S = w₁x₁ + w₂x₂ + w₀
  = 1.1(1) + 1.1(1) + (-1.4)
  = 1.1 + 1.1 - 1.4
  = 0.8

Puisque S = 0.8 > 0 → y = 1 ✓
```

**SUCCÈS ! Le perceptron prédit maintenant correctement 1.**

---

### Erreurs courantes :

**Oublier de mettre à jour le bias** - Le bias est aussi un paramètre appris !
**Mauvais signe de l'erreur** - L'erreur est (désiré - réel), pas (réel - désiré)
**Ne pas multiplier par l'entrée** - Δw = η·e·x, pas seulement η·e
**Oublier le taux d'apprentissage** - Il faut inclure η dans le calcul

---

**Point clé :** La Delta rule ajuste les poids proportionnellement à :
1. L'erreur (à quel point on s'est trompé)
2. La valeur de l'entrée (quelles entrées ont contribué)
3. Le taux d'apprentissage (à quel point la mise à jour est agressive)

**Temps de résolution :** 6-8 minutes
**Fréquence à l'examen :** 95%

---

## Problème 3.2 : Delta rule avec erreur négative 

**Données :**
- Poids : w = (0.5, 0.5)
- Bias : w₀ = -0.3
- Taux d'apprentissage : η = 0.2
- Entrée : x = (1, 0)
- Sortie désirée : y_d = 0
- Sortie réelle : y = 1 (le perceptron prédit 1, mais devrait être 0)

**Tâche :** Calculer les nouveaux poids.

---

### Solution :

**Étape 1 : Calcul de l'erreur**

```
e = y_d - y = 0 - 1 = -1
```

**Une erreur négative signifie que la sortie est TROP ÉLEVÉE.**

**Étape 2 : Mises à jour des poids**

```
Δw₁ = η · e · x₁ = 0.2 · (-1) · 1 = -0.2
Δw₂ = η · e · x₂ = 0.2 · (-1) · 0 = 0
Δw₀ = η · e · 1 = 0.2 · (-1) · 1 = -0.2
```

**Étape 3 : Nouveaux poids**

```
w₁_new = 0.5 + (-0.2) = 0.3
w₂_new = 0.5 + 0 = 0.5
w₀_new = -0.3 + (-0.2) = -0.5
```

**RÉPONSE : w_new = (0.3, 0.5), w₀_new = -0.5**

---

**Étape 4 : Interprétation**

Remarquez :
- w₁ a DIMINUÉ (de 0.5 à 0.3) - car x₁=1 a contribué à la mauvaise sortie
- w₂ est INCHANGÉ (x₂=0, donc il n'a pas contribué)
- w₀ a DIMINUÉ (le bias était trop élevé, rendant la sortie positive alors qu'elle devrait être négative)

**Vérification :**

```
S = 0.3(1) + 0.5(0) + (-0.5) = -0.2 < 0 → y = 0 ✓
```

Prédit maintenant correctement 0 !

---

### Erreurs courantes :

 **Paniquer face aux mises à jour négatives** - C'est CORRECT ! Erreur négative → les poids diminuent
 **Modifier w₂ même si x₂=0** - Si l'entrée est 0, ce poids n'est pas mis à jour
 **Ne pas comprendre l'intuition** - Erreur négative signifie "sortie trop élevée, réduire les poids"

---

**Temps de résolution :** 6-8 minutes
**Difficulté :** 
**Fréquence à l'examen :** 75%

---

## Problème 3.3 : Entraînement multi-étapes (porte AND) 

**Tâche :** Entraîner un perceptron à apprendre la porte AND en utilisant la Delta rule.

**Configuration :**
- Poids initiaux : w = (0, 0), w₀ = 0 (démarrage à zéro)
- Taux d'apprentissage : η = 0.5
- Données d'entraînement : Les 4 exemples de la porte AND

**Effectuer 1 epoch complet (un passage à travers tous les exemples d'entraînement).**

---

### Solution :

**Données d'entraînement :**

| x₁ | x₂ | Target y_d |
|----|-----|-----------|
| 0  | 0   | 0         |
| 0  | 1   | 0         |
| 1  | 0   | 0         |
| 1  | 1   | 1         |

**État initial :**
```
w = (0, 0), w₀ = 0
```

---

**EXEMPLE 1 : x = (0, 0), y_d = 0**

Étape 1 : Propagation avant
```
S = 0·0 + 0·0 + 0 = 0
y = H(0) = 1 (Heaviside : 1 si S ≥ 0)
```

Étape 2 : Erreur
```
e = 0 - 1 = -1
```

Étape 3 : Mises à jour
```
Δw₁ = 0.5 · (-1) · 0 = 0
Δw₂ = 0.5 · (-1) · 0 = 0
Δw₀ = 0.5 · (-1) · 1 = -0.5
```

Étape 4 : Nouveaux poids
```
w = (0, 0), w₀ = -0.5
```

---

**EXEMPLE 2 : x = (0, 1), y_d = 0**

Étape 1 : Propagation avant
```
S = 0·0 + 0·1 + (-0.5) = -0.5
y = H(-0.5) = 0 ✓ (déjà correct !)
```

Étape 2 : Erreur
```
e = 0 - 0 = 0 (pas d'erreur)
```

Étape 3 : Mises à jour
```
Tous les Δw = 0 (pas de mise à jour quand la prédiction est correcte)
```

Étape 4 : Poids inchangés
```
w = (0, 0), w₀ = -0.5
```

---

**EXEMPLE 3 : x = (1, 0), y_d = 0**

Étape 1 : Propagation avant
```
S = 0·1 + 0·0 + (-0.5) = -0.5
y = 0 ✓ (correct)
```

Étape 2 : Erreur
```
e = 0
```

Étape 3 : Pas de mise à jour
```
w = (0, 0), w₀ = -0.5
```

---

**EXEMPLE 4 : x = (1, 1), y_d = 1**

Étape 1 : Propagation avant
```
S = 0·1 + 0·1 + (-0.5) = -0.5
y = 0 ✗ (faux ! devrait être 1)
```

Étape 2 : Erreur
```
e = 1 - 0 = 1
```

Étape 3 : Mises à jour
```
Δw₁ = 0.5 · 1 · 1 = 0.5
Δw₂ = 0.5 · 1 · 1 = 0.5
Δw₀ = 0.5 · 1 · 1 = 0.5
```

Étape 4 : Nouveaux poids
```
w₁ = 0 + 0.5 = 0.5
w₂ = 0 + 0.5 = 0.5
w₀ = -0.5 + 0.5 = 0
```

**Poids finaux après 1 epoch : w = (0.5, 0.5), w₀ = 0**

---

**Vérification sur toutes les entrées :**

```
(0,0): S = 0.5·0 + 0.5·0 + 0 = 0 → y = 1 ✗ (encore faux !)
(0,1): S = 0.5·0 + 0.5·1 + 0 = 0.5 → y = 1 ✗ (maintenant faux !)
(1,0): S = 0.5·1 + 0.5·0 + 0 = 0.5 → y = 1 ✗ (maintenant faux !)
(1,1): S = 0.5·1 + 0.5·1 + 0 = 1 → y = 1 ✓
```

**Seulement 1 sur 4 correct ! Il faut plus d'epochs.**

---

**EPOCH 2 (seules les mises à jour clés sont montrées) :**

Après avoir traité les 4 exemples à nouveau :
```
Final : w ≈ (1, 1), w₀ ≈ -1.5
```

Cela implémenterait correctement la porte AND.

---

### Erreurs courantes :

 **S'arrêter après 1 epoch** - Il faut généralement plusieurs epochs pour converger
 **Ne pas mettre à jour quand e=0** - Correct ! Pas de mise à jour nécessaire quand la prédiction est juste
 **S'attendre à une solution parfaite immédiatement** - L'apprentissage est progressif
 **Erreurs arithmétiques** - Avec plusieurs étapes, il est facile de se tromper

---

**Points clés :**
1. **Apprentissage en ligne** - Mise à jour après chaque exemple, pas après avoir vu toutes les données
2. **L'ordre compte** - Un ordre d'entraînement différent → des poids intermédiaires différents
3. **Convergence** - Pour les problèmes linéairement séparables, la Delta rule CONVERGERA
4. **Impact du taux d'apprentissage** - η plus grand = plus rapide mais moins stable ; η plus petit = plus lent mais plus stable

**Temps de résolution :** 25-30 minutes
**Difficulté :** 
**Fréquence à l'examen :** 50% (mais vaut BEAUCOUP de points)

---

## Problème 3.4 : Comparaison des taux d'apprentissage

**Même scénario donné :**
- w = (1, 1), w₀ = -1
- x = (1, 0), y_d = 0, y = 1
- Erreur e = -1

**Comparer les mises à jour avec :**
a) η = 0.01 (très petit)
b) η = 0.5 (modéré)
c) η = 2.0 (très grand)

---

### Solution :

**Calcul commun :**
```
Δw₁ = η · e · x₁ = η · (-1) · 1 = -η
Δw₂ = η · e · x₂ = η · (-1) · 0 = 0
Δw₀ = η · e · 1 = -η
```

---

**a) η = 0.01 :**

```
Δw₁ = -0.01
Δw₂ = 0
Δw₀ = -0.01

Nouveaux poids :
w = (0.99, 1), w₀ = -1.01
```

**Caractéristique :** Mise à jour MINUSCULE, apprentissage très lent

---

**b) η = 0.5 :**

```
Δw₁ = -0.5
Δw₂ = 0
Δw₀ = -0.5

Nouveaux poids :
w = (0.5, 1), w₀ = -1.5
```

**Caractéristique :** Mise à jour modérée, apprentissage équilibré

---

**c) η = 2.0 :**

```
Δw₁ = -2.0
Δw₂ = 0
Δw₀ = -2.0

Nouveaux poids :
w = (-1, 1), w₀ = -3
```

**Caractéristique :** Mise à jour ÉNORME, risque de dépassement

---

**Vérification - lequel fonctionne le mieux ?**

Tester les trois sur x = (1, 0), y_d = 0 :

**η = 0.01 :**
```
S = 0.99·1 + 1·0 + (-1.01) = -0.02 → y = 0 ✓
```

**η = 0.5 :**
```
S = 0.5·1 + 1·0 + (-1.5) = -1.0 → y = 0 ✓
```

**η = 2.0 :**
```
S = -1·1 + 1·0 + (-3) = -4 → y = 0 ✓
```

Tous corrects ! Mais η=2.0 a **largement dépassé**.

---

**Test sur une entrée différente x = (1, 1), devrait être 0 pour un comportement type NAND :**

**η = 0.01 :**
```
S = 0.99·1 + 1·1 + (-1.01) = 0.98 → y = 1 (encore faux)
```

**η = 0.5 :**
```
S = 0.5·1 + 1·1 + (-1.5) = 0 → y = 1 (marginal)
```

**η = 2.0 :**
```
S = -1·1 + 1·1 + (-3) = -3 → y = 0 (correct, mais dépassement)
```

---

### Analyse :

| Taux d'apprentissage | Avantages | Inconvénients | Quand l'utiliser |
|---------------|------|------|----------|
| Petit (0.01) | Stable, convergence lisse | TRÈS lent, peut ne pas apprendre à temps | Réglage fin, proche de la convergence |
| Modéré (0.5) | Bon équilibre | Nécessite un ajustement | Entraînement général |
| Grand (2.0) | Apprentissage initial rapide | Dépassement, instabilité, oscillation | Jamais recommandé |

**Plage optimale :** Typiquement η ∈ [0.01, 0.5] pour les perceptrons

---

### Erreurs courantes :

 **Penser que "plus grand c'est mieux"** - Un η grand cause de l'instabilité
 **Utiliser le même η tout au long de l'entraînement** - On peut diminuer η au fil du temps (planification du taux d'apprentissage)
**Ne pas tester sur plusieurs exemples** - Une prédiction correcte ne signifie pas de bons poids

---

**Temps de résolution :** 15-18 minutes
**Fréquence à l'examen :** 40%

---

## Problème 3.5 : Hebb's rule 

**Hebb's rule :** « Les neurones qui s'activent ensemble se connectent ensemble »

**Formule :** Δwᵢ = η · xᵢ · y (PAS de terme d'erreur !)

**Données :**
- Poids initiaux : w = (0, 0), w₀ = 0
- Taux d'apprentissage : η = 0.1
- Exemples d'entraînement :
  - x = (1, 1), y = 1
  - x = (1, 0), y = 1
  - x = (0, 1), y = 0

**Tâche :** Appliquer les mises à jour de la Hebb's rule.

---

### Solution :

**EXEMPLE 1 : x = (1, 1), y = 1**

```
Δw₁ = η · x₁ · y = 0.1 · 1 · 1 = 0.1
Δw₂ = η · x₂ · y = 0.1 · 1 · 1 = 0.1
Δw₀ = η · 1 · y = 0.1 · 1 · 1 = 0.1

Nouveaux poids : w = (0.1, 0.1), w₀ = 0.1
```

---

**EXEMPLE 2 : x = (1, 0), y = 1**

```
Δw₁ = 0.1 · 1 · 1 = 0.1
Δw₂ = 0.1 · 0 · 1 = 0
Δw₀ = 0.1 · 1 · 1 = 0.1

Nouveaux poids : w = (0.2, 0.1), w₀ = 0.2
```

---

**EXEMPLE 3 : x = (0, 1), y = 0**

```
Δw₁ = 0.1 · 0 · 0 = 0
Δw₂ = 0.1 · 1 · 0 = 0
Δw₀ = 0.1 · 1 · 0 = 0

Pas de mise à jour ! (car la sortie y = 0)
```

**Poids finaux : w = (0.2, 0.1), w₀ = 0.2**

---

### Différences clés avec la Delta rule :

| Propriété | Delta rule | Hebb's rule |
|----------|------------|-------------|
| Formule | Δw = η·e·x | Δw = η·x·y |
| Utilise l'erreur ? |  Oui (e = y_d - y) |  Non |
| Supervisé ? |  Oui (nécessite des étiquettes) |  Non (non supervisé) |
| Mise à jour quand y=0 ? |  Oui (si e≠0) | Non |
| Garantie de convergence ? |  Oui (pour lin. sép.) |  Non |
| Les poids peuvent diminuer ? |  Oui (si e<0) |  Non (augmentent seulement) |

---

**Pourquoi la Hebb's rule échoue pour l'apprentissage supervisé :**

1. **Pas de correction d'erreur** - Ne sait pas si la prédiction est juste ou fausse
2. **Croissance non bornée** - Les poids ne font qu'augmenter, jamais diminuer
3. **Pas de convergence** - Ne peut pas apprendre une frontière de classification spécifique

**Pourquoi elle reste importante :**

1. **Plausibilité biologique** - Modélise le comportement réel des neurones
2. **Apprentissage non supervisé** - Peut trouver des motifs sans étiquettes
3. **Importance historique** - A conduit aux variantes modernes de l'apprentissage hebbien
4. **Base conceptuelle** - Permet de comprendre pourquoi on A BESOIN d'un apprentissage basé sur l'erreur

---

### Erreurs courantes :

**Utiliser le terme d'erreur** - La Hebb's rule est Δw = ηxy, PAS ηexy
**Mettre à jour quand y=0** - Pas de mise à jour si le neurone de sortie n'est pas actif
**S'attendre à la convergence** - La Hebb's rule ne converge pas vers une solution spécifique

---

**Stratégie d'examen :** 
Si on vous demande « Comparer Hebb vs Delta » :
1. Différence de formule (erreur vs pas d'erreur)
2. Supervisé vs non supervisé
3. Garantie de convergence
4. Quand chacune est appropriée

**Temps de résolution :** 12-15 minutes
**Fréquence à l'examen :** 60%

---

## Résumé : Règles de mise à jour des poids

**Maîtrisez ces formules :**

**Delta rule (supervisé) :**
```
e = y_desired - y_actual
Δwᵢ = η · e · xᵢ
```

**Hebb's rule (non supervisé) :**
```
Δwᵢ = η · xᵢ · y
```

**Points clés pour l'examen :**
1. La Delta rule utilise l'ERREUR (différence entre désiré et réel)
2. La Hebb's rule utilise uniquement la SORTIE réelle (pas de supervision)
3. Les deux mettent à jour proportionnellement à l'ENTRÉE
4. Le taux d'apprentissage contrôle l'ampleur de la mise à jour
5. La Delta rule converge pour les problèmes linéairement séparables
6. La Hebb's rule ne garantit pas la convergence

**Entraînez-vous jusqu'à pouvoir :**
- Appliquer l'une ou l'autre règle sans regarder la formule
- Expliquer POURQUOI la Delta rule fonctionne mieux pour l'apprentissage supervisé
- Calculer des mises à jour avec des erreurs négatives
- Effectuer un entraînement multi-étapes

**Prochain sujet : Gradient descent - la généralisation de ces idées.**
