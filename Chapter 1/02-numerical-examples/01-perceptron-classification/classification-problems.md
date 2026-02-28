# Problèmes de Classification avec le Perceptron

## Problème 1.1 : Porte AND de base

**Données :**
- Poids : w = (1, 1)
- Bias : w₀ = -1.5
- Activation : Heaviside (fonction échelon)
- Entrées à classifier : (0,0), (0,1), (1,0), (1,1)

**Travail demandé :** 
a) Calculer la somme pondérée pour chaque entrée
b) Appliquer la fonction d'activation
c) Déterminer la sortie (0 ou 1)
d) Vérifier que cela implémente la logique AND

---

### Solution :

**Étape 1 : Rappeler l'équation du perceptron**

Pour 2 entrées :
```
S = w₁x₁ + w₂x₂ + w₀
y = H(S) où H(S) = {1 si S ≥ 0, 0 si S < 0}
```

**Étape 2 : Calculer pour chaque entrée**

**Entrée (0, 0) :**
```
S = 1(0) + 1(0) + (-1.5) = -1.5
Puisque S = -1.5 < 0 → y = 0
```

**Entrée (0, 1) :**
```
S = 1(0) + 1(1) + (-1.5) = -0.5
Puisque S = -0.5 < 0 → y = 0
```

**Entrée (1, 0) :**
```
S = 1(1) + 1(0) + (-1.5) = -0.5
Puisque S = -0.5 < 0 → y = 0
```

**Entrée (1, 1) :**
```
S = 1(1) + 1(1) + (-1.5) = 0.5
Puisque S = 0.5 > 0 → y = 1
```

**Étape 3 : Vérifier la logique AND**

| x₁ | x₂ | AND | Sortie du Perceptron| ✓/✗|
|----|----|----|-------------------|-----|
| 0  | 0  | 0  | 0                 | ✓   |
| 0  | 1  | 0  | 0                 | ✓   |
| 1  | 0  | 0  | 0                 | ✓   |
| 1  | 1  | 1  | 1                 | ✓   |

**VÉRIFIÉ : Ce perceptron implémente correctement la logique AND.**

**Étape 4 : Frontière de décision**

La frontière de décision est là où S = 0 :
```
w₁x₁ + w₂x₂ + w₀ = 0
1·x₁ + 1·x₂ - 1.5 = 0
x₂ = 1.5 - x₁
```

C'est une droite de pente -1 et d'ordonnée à l'origine 1.5.
- Les points AU-DESSUS de la droite (x₁ + x₂ > 1.5) → sortie 1
- Les points EN-DESSOUS de la droite (x₁ + x₂ < 1.5) → sortie 0

---

### Erreurs courantes :

 **Oublier le terme de bias** - Beaucoup d'étudiants calculent S = w₁x₁ + w₂x₂ uniquement
 **Mauvais signe du bias** - w₀ = -1.5, pas +1.5
 **Confusion sur le seuil** - Heaviside s'active à S ≥ 0, pas S > 0
 **Ne pas vérifier toutes les entrées** - Il faut vérifier les 4 combinaisons

---

**Conseils d'examen :**
- Toujours écrire la formule en premier
- Montrer le calcul pour chaque entrée étape par étape
- Créer la table de vérité pour vérifier
- Ce problème exact apparaît dans 90% des examens de DL

**Temps de résolution :** 5-7 minutes
**Fréquence d'examen :** 95%

---

## Problème 1.2 : Implémentation de la porte OR

**Travail demandé :** Trouver les poids (w₁, w₂) et le bias w₀ qui implémentent la porte OR.

**Table de vérité OR :**
| x₁ | x₂ | OR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 1   |

---

### Solution :

**Étape 1 : Raisonnement**

Pour la porte OR :
- La sortie doit être 1 si AU MOINS UNE entrée est 1
- La sortie doit être 0 uniquement quand les DEUX entrées sont 0

**Étape 2 : Choisir les poids**

Essayons : w = (1, 1), w₀ = -0.5

**Raisonnement :** 
- Il faut S > 0 quand x₁=1 OU x₂=1
- Il faut S < 0 uniquement quand les deux sont 0
- Un bias de -0.5 signifie qu'il faut que la somme des entrées > 0.5

**Étape 3 : Vérifier**

**Entrée (0, 0) :**
```
S = 1(0) + 1(0) - 0.5 = -0.5 < 0 → y = 0 ✓
```

**Entrée (0, 1) :**
```
S = 1(0) + 1(1) - 0.5 = 0.5 > 0 → y = 1 ✓
```

**Entrée (1, 0) :**
```
S = 1(1) + 1(0) - 0.5 = 0.5 > 0 → y = 1 ✓
```

**Entrée (1, 1) :**
```
S = 1(1) + 1(1) - 0.5 = 1.5 > 0 → y = 1 ✓
```

**RÉPONSE : w = (1, 1), w₀ = -0.5 implémente la porte OR**

**Étape 4 : Solutions alternatives**

Remarque : De nombreuses solutions fonctionnent ! Par exemple :
- w = (2, 2), w₀ = -1 (la mise à l'échelle ne change pas le comportement)
- w = (1, 1), w₀ = -0.1 (tout bias entre -1 et 0)
- w = (0.5, 0.5), w₀ = -0.25

**Point clé :** Tant que :
- w₁ + w₂ + w₀ > 0 (pour les entrées (1,1), (1,0), (0,1))
- w₀ < 0 (pour l'entrée (0,0))

---

### Erreurs courantes :

 **Utiliser les poids de la porte AND** - Facile à confondre avec le Problème 1.1
 **Ne pas vérifier les 4 entrées** - Il faut vérifier chaque cas
 **Oublier que plusieurs solutions existent** - Il n'y a pas une seule réponse "correcte"

---

**Temps de résolution :** 8-10 minutes
**Fréquence d'examen :** 75%

---

## Problème 1.3 : Porte NAND (AND inversé)

**Données :** Implémenter la porte NAND en utilisant un perceptron avec activation Heaviside.

**Table de vérité NAND :**
| x₁ | x₂ | NAND |
|----|-----|------|
| 0  | 0  | 1    |
| 0  | 1  | 1    |
| 1  | 0  | 1    |
| 1  | 1  | 0    |

**Indication :** NAND est NOT-AND (porte AND inversée)

---

### Solution :

**Étape 1 : Intuition**

NAND est l'opposé de AND. Deux approches :
1. Inverser les poids de la porte AND
2. Inverser le bias pour changer la frontière de décision

**Étape 2 : Méthode - Inverser les poids de la porte AND**

La porte AND était : w = (1, 1), w₀ = -1.5

Pour NAND, essayons : w = (-1, -1), w₀ = 1.5

**Raisonnement :**
- Les poids négatifs signifient que les entrées DIMINUENT la somme
- Un bias positif signifie que la sortie par défaut est positive
- Ce n'est que quand les DEUX entrées sont 1 que la somme descend en dessous de 0

**Étape 3 : Vérifier**

**Entrée (0, 0) :**
```
S = -1(0) + -1(0) + 1.5 = 1.5 > 0 → y = 1 ✓
```

**Entrée (0, 1) :**
```
S = -1(0) + -1(1) + 1.5 = 0.5 > 0 → y = 1 ✓
```

**Entrée (1, 0) :**
```
S = -1(1) + -1(0) + 1.5 = 0.5 > 0 → y = 1 ✓
```

**Entrée (1, 1) :**
```
S = -1(1) + -1(1) + 1.5 = -0.5 < 0 → y = 0 ✓
```

**RÉPONSE : w = (-1, -1), w₀ = 1.5 implémente NAND**

**Étape 4 : Frontière de décision**

```
-x₁ - x₂ + 1.5 = 0
x₂ = 1.5 - x₁
```

Même droite que AND, mais les régions sont INVERSÉES !
- AND : la région AU-DESSUS de la droite donne 1
- NAND : la région EN-DESSOUS de la droite donne 1

---

### Erreurs courantes :

 **Oublier d'inverser TOUS les poids** - Il faut inverser à la fois w₁ et w₂
 **Mauvais signe du bias** - Doit être +1.5, pas -1.5
 **Confusion avec la porte NOR** - NAND ≠ NOR

---

**Concept clé :** Les transformations linéaires (inversion des poids) inversent les régions de décision.

**Temps de résolution :** 8-10 minutes
**Difficulté :**
**Fréquence d'examen :** 60%

---

## Problème 1.4 : Impossibilité du XOR

**Démontrer :** Un perceptron simple avec 2 entrées NE PEUT PAS implémenter la fonction XOR.

**Table de vérité XOR :**
| x₁ | x₂ | XOR |
|----|-----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

---

### Solution :

**Étape 1 : Ce que XOR exige**

XOR donne 1 quand les entrées sont DIFFÉRENTES :
- (0,1) → 1
- (1,0) → 1

XOR donne 0 quand les entrées sont IDENTIQUES :
- (0,0) → 0
- (1,1) → 0

**Étape 2 : Exigence de séparabilité linéaire**

Un perceptron ne peut résoudre que des problèmes LINÉAIREMENT SÉPARABLES.

**Définition :** Un problème est linéairement séparable si l'on peut tracer une DROITE qui sépare les deux classes.

Pour XOR :
- Classe 1 (sortie=1) : points (0,1) et (1,0)
- Classe 0 (sortie=0) : points (0,0) et (1,1)

**Étape 3 : Preuve géométrique**

Tracer les points sur le plan 2D :

```
  x₂
  1  |  (0,1)[1]    (1,1)[0]
     |
  0  |  (0,0)[0]    (1,0)[1]
     |________________
        0            1     x₁
```

Où [1] signifie sortie=1, [0] signifie sortie=0.

**Essayez de tracer une droite séparant les 1 des 0 :**

Peu importe où vous tracez la droite :
- Si la droite sépare (0,1) de (1,1), elle sépare AUSSI (1,0) de (0,0) incorrectement
- Si la droite sépare (1,0) de (0,0), elle sépare AUSSI (0,1) de (1,1) incorrectement

Les points de sortie=1 sont DIAGONALEMENT OPPOSÉS !
Les points de sortie=0 sont aussi DIAGONALEMENT OPPOSÉS !

**Aucune droite ne peut les séparer.**

**Étape 4 : Preuve algébrique**

Supposons qu'il existe des poids w₁, w₂, w₀ qui résolvent XOR :

De (0,0) → 0 : 
```
w₀ < 0  ... (équation 1)
```

De (0,1) → 1 :
```
w₂ + w₀ > 0  ... (équation 2)
```

De (1,0) → 1 :
```
w₁ + w₀ > 0  ... (équation 3)
```

De (1,1) → 0 :
```
w₁ + w₂ + w₀ < 0  ... (équation 4)
```

**Des équations (2) et (3) :**
```
w₂ + w₀ > 0  →  w₂ > -w₀
w₁ + w₀ > 0  →  w₁ > -w₀

En additionnant :
w₁ + w₂ > -2w₀

Par conséquent :
w₁ + w₂ + w₀ > w₀ - w₀ = 0

Donc : w₁ + w₂ + w₀ > 0
```

**Mais l'équation (4) exige :** w₁ + w₂ + w₀ < 0

**CONTRADICTION ! Aucune solution n'existe.**

---

### Erreurs courantes :

 **Ne pas dessiner le diagramme géométrique** - La preuve visuelle est la plus claire
 **Essayer des poids au hasard** - La preuve algébrique montre que c'est impossible, pas juste difficile
 **Confondre avec "difficile"** - Ce n'est pas difficile, c'est IMPOSSIBLE pour un perceptron simple

---

**Point clé :** 
- Les perceptrons linéaires ne peuvent résoudre que des problèmes linéairement séparables
- XOR est l'exemple canonique de problème NON linéairement séparable
- Il faut un réseau multicouche (MLP) pour résoudre XOR

**Pourquoi c'est important pour l'examen :**
- Montre que vous comprenez les LIMITES du perceptron
- Teste à la fois le raisonnement géométrique et algébrique
- Introduit le besoin des réseaux multicouches (Chapitre 2 !)

**Temps de résolution :** 15-20 minutes
**Fréquence d'examen :** 60% (valeur élevée en points)

---

## Problème 1.5 : Perceptron à 3 entrées

**Données :**
- Poids : w = (1, 2, -1)
- Bias : w₀ = 0
- Activation : Heaviside
- Entrées à classifier : (1,0,1), (0,1,0), (1,1,1), (0,0,1)

**Travail demandé :** Classifier chaque entrée.

---

### Solution :

**Étape 1 : Formule du perceptron pour 3 entrées**

```
S = w₁x₁ + w₂x₂ + w₃x₃ + w₀
y = H(S)
```

**Étape 2 : Classifier chaque entrée**

**Entrée (1, 0, 1) :**
```
S = 1(1) + 2(0) + (-1)(1) + 0
S = 1 + 0 - 1 + 0 = 0
Puisque S = 0 ≥ 0 → y = 1
```

**Entrée (0, 1, 0) :**
```
S = 1(0) + 2(1) + (-1)(0) + 0
S = 0 + 2 + 0 + 0 = 2
Puisque S = 2 > 0 → y = 1
```

**Entrée (1, 1, 1) :**
```
S = 1(1) + 2(1) + (-1)(1) + 0
S = 1 + 2 - 1 + 0 = 2
Puisque S = 2 > 0 → y = 1
```

**Entrée (0, 0, 1) :**
```
S = 1(0) + 2(0) + (-1)(1) + 0
S = 0 + 0 - 1 + 0 = -1
Puisque S = -1 < 0 → y = 0
```

**Récapitulatif :**

| Entrée    | S  | Sortie |
|-----------|-----|--------|
| (1,0,1)   | 0   | 1      |
| (0,1,0)   | 2   | 1      |
| (1,1,1)   | 2   | 1      |
| (0,0,1)   | -1  | 0      |

**Étape 3 : Interprétation**

Remarques :
- L'entrée x₂ a un poids de 2 (influence la PLUS FORTE)
- L'entrée x₃ a un poids de -1 (influence NÉGATIVE)
- Si x₂=1, la sortie est probablement 1 (le poids est grand et positif)
- Si seul x₃=1, la sortie est 0 (le poids négatif tire la somme vers le bas)

---

### Erreurs courantes :

 **Erreurs de calcul** - Avec 3 termes, les erreurs arithmétiques sont faciles à faire
 **Oublier le bias** - Même si w₀=0, il faut l'inclure dans la formule
 **Cas limite S=0** - Heaviside s'active à S≥0, pas S>0

---

**Concept clé :** 
- Plus de dimensions = plus difficile à visualiser
- La frontière de décision est maintenant un PLAN dans l'espace 3D : x₁ + 2x₂ - x₃ = 0
- Mêmes principes qu'en 2D, simplement étendus

**Temps de résolution :** 10-12 minutes
**Fréquence d'examen :** 50%

---

## Problème 1.6 : Trouver les poids à partir des exigences

**Travail demandé :** Concevoir un perceptron qui donne 1 si et seulement si :
- x₁ + x₂ ≥ 3

Trouver les poids et le bias appropriés.

---

### Solution :

**Étape 1 : Traduire l'exigence en perceptron**

On veut : 
- Sortie = 1 quand x₁ + x₂ ≥ 3
- Sortie = 0 quand x₁ + x₂ < 3

Frontière de décision : x₁ + x₂ = 3

**Étape 2 : Faire correspondre à l'équation du perceptron**

Le perceptron s'active quand : w₁x₁ + w₂x₂ + w₀ ≥ 0

On veut que cela soit équivalent à : x₁ + x₂ ≥ 3

Réarrangeons notre exigence :
```
x₁ + x₂ ≥ 3
x₁ + x₂ - 3 ≥ 0
```

**Faire correspondre les termes :**
```
w₁x₁ + w₂x₂ + w₀ ≥ 0
  ↓      ↓      ↓
  1x₁ +  1x₂ +  (-3) ≥ 0
```

**Réponse : w = (1, 1), w₀ = -3**

**Étape 3 : Vérifier avec des cas de test**

**Test x=(2, 0.5) :** Devrait donner 0 (puisque 2+0.5=2.5 < 3)
```
S = 1(2) + 1(0.5) - 3 = -0.5 < 0 → y = 0 ✓
```

**Test x=(2, 2) :** Devrait donner 1 (puisque 2+2=4 ≥ 3)
```
S = 1(2) + 1(2) - 3 = 1 > 0 → y = 1 ✓
```

**Test x=(1.5, 1.5) :** Devrait donner 1 (puisque 1.5+1.5=3 ≥ 3)
```
S = 1(1.5) + 1(1.5) - 3 = 0 ≥ 0 → y = 1 ✓
```

---

### Erreurs courantes :

 **Mauvais signe du bias** - Facile d'écrire w₀ = 3 au lieu de -3
 **Ne pas vérifier** - Toujours tester les cas limites
 **Trop réfléchir** - Traduction directe de l'exigence vers les poids

---

**Stratégie d'examen :**
Ce type de problème de "rétro-ingénierie" est courant :
1. Identifier la frontière de décision à partir de l'exigence
2. Réarranger pour obtenir la forme ≥ 0
3. Lire directement les poids
4. Vérifier avec des cas de test

**Temps de résolution :** 8-10 minutes
**Fréquence d'examen :** 45%

---

## Résumé & Stratégie de révision

**Pour la préparation à l'examen :**

1. **Maîtriser le Problème 1.1** (porte AND) - C'est la BASE
2. **Comprendre le Problème 1.4** (impossibilité du XOR) - Teste la compréhension profonde
3. **Pratiquer les Problèmes 1.2, 1.3** - Développer l'aisance avec les différentes portes logiques
4. **Se challenger avec 1.5, 1.6** - Raisonnement de niveau supérieur

**Répartition du temps :**
- Consacrer 60% du temps aux Problèmes 1.1, 1.2, 1.3 (haute fréquence, difficulté moyenne)
- Consacrer 30% au Problème 1.4 (fréquence plus basse mais valeur élevée en points)
- Consacrer 10% aux Problèmes 1.5, 1.6 (si le temps le permet)

**Prochaines étapes :**
1. Tenter les 6 problèmes SANS regarder les solutions
2. Vérifier votre travail, identifier les erreurs
3. Refaire les problèmes mal résolus
4. Passer au dossier des fonctions d'activation

---
