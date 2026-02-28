# QUESTIONS DE COURS — Chapitre 1 : Concepts Fondamentaux d'un Réseau de Neurones

---

## SECTION 1 — Machine Learning et Deep Learning

**Q1. Qu'est-ce que le Machine Learning (ML) ?**
→ Le Machine Learning est un ensemble de méthodes permettant de détecter automatiquement des régularités dans des données. C'est un domaine de l'intelligence artificielle (IA). Chaque méthode est associée à un algorithme d'optimisation spécifique (ex : descente de gradient pour les modèles linéaires, CART pour les arbres de décision, marge maximum pour les SVM).

**Q2. Quelles sont les deux formes principales du Machine Learning ?**
→ L'apprentissage **supervisé** et l'apprentissage **non supervisé**.

**Q3. Qu'est-ce que l'apprentissage supervisé ?**
→ L'apprentissage supervisé correspond aux situations où les données sont **étiquetées** : pour chaque exemple d'entrée, on dispose de la réponse correcte (la sortie désirée). L'objectif est d'apprendre une règle générale permettant de prédire la sortie pour une nouvelle donnée inconnue.

**Q4. Comment se note une base d'entraînement supervisée ?**
→ Une base d'entraînement contenant N exemples étiquetés se note :
$$B = (X_i, y_i^d)_{1 \leq i \leq N}$$
où $X_i$ est le vecteur d'entrée de dimension $n$ et $y_i^d$ est la sortie désirée associée.

**Q5. Que représentent $X_i$, $(x_{i1}, \ldots, x_{in})$ et $n$ dans la notation de la base ?**
→ $X_i$ est le vecteur d'entrée associé à l'exemple $i$. $(x_{i1}, \ldots, x_{in})$ sont les attributs ou **caractéristiques** (*features*) décrivant l'exemple. $n$ est le nombre total de caractéristiques pour chaque donnée.

**Q6. Quelle est la différence entre classification et régression ?**
→
- **Classification** : la sortie $y_i^d$ représente une classe (catégorie), typiquement un entier dans un ensemble fini $y_i^d \in \{1, \ldots, C\}$. L'objectif est d'associer chaque entrée à une catégorie.
- **Régression** : la sortie $y_i^d$ est un nombre réel. L'objectif est de prédire une valeur numérique continue.

**Q7. Donnez deux exemples de problèmes de régression.**
→
- Météorologie : $X$ = altitude d'une station, $y$ = température mesurée.
- Santé : $X$ = âge d'un individu, $y$ = tension artérielle.

**Q8. Donnez deux exemples de problèmes de classification.**
→
- Classification d'images d'animaux : $X$ = pixels de l'image, $y$ = espèce de l'animal.
- Analyse de sentiments : $X$ = phrase ou avis, $y$ = sentiment (joie, peur, colère...).

**Q9. Qu'est-ce que l'apprentissage non supervisé ?**
→ L'apprentissage non supervisé correspond aux situations où les données sont **non étiquetées**. On ne connaît pas la réponse ou la catégorie associée à chaque exemple. On dispose uniquement des entrées $B = (X_i)_{1 \leq i \leq N}$. L'objectif est de découvrir automatiquement des structures cachées ou des régularités dans les données.

**Q10. Qu'est-ce que le clustering ?**
→ Le clustering est une tâche courante en apprentissage non supervisé qui consiste à regrouper les données en sous-ensembles appelés **clusters**. Les données d'un même cluster sont similaires selon un critère choisi, et les différents clusters représentent des comportements ou profils distincts.

**Q11. Quelles sont les trois types de tâches en apprentissage non supervisé ?**
→
1. **Clustering** : regrouper des données en familles homogènes.
2. **Réduction de dimension** : simplifier les données tout en conservant l'essentiel (exemple : ACP/PCA).
3. **Détection d'anomalies** : trouver des données très différentes des autres.

**Q12. Qu'est-ce qu'un modèle de ML ? Comment est-il défini ?**
→ Un modèle de ML est essentiellement une **fonction mathématique** $f(x)$ qui prend une donnée en entrée et produit une prédiction en sortie. L'objectif du ML est de trouver les **paramètres optimaux** de ce modèle (poids, coefficients) en les ajustant progressivement pour améliorer les prédictions.

**Q13. Quel est le rôle de la fonction coût ?**
→ La fonction coût (ou fonction de perte) mesure l'erreur commise par le modèle en comparant les prédictions $f(x)$ aux valeurs réelles $y$ du dataset. Une petite valeur = bonnes prédictions. Une grande valeur = mauvaises prédictions. Elle joue le rôle de « boussole » pour guider l'apprentissage.

**Q14. Qu'est-ce que la descente de gradient et quel est son rôle en ML ?**
→ La descente de gradient est l'algorithme d'optimisation le plus répandu en ML. C'est un processus itératif où : (1) on calcule la direction dans laquelle la fonction coût diminue le plus vite, (2) on met à jour les paramètres dans cette direction, (3) on répète jusqu'à obtenir un modèle performant.

**Q15. Quelle est la différence entre ML classique et Deep Learning ?**
→ Le ML classique repose souvent sur un **modèle unique** pour effectuer des prédictions. Le DL utilise des **réseaux de neurones artificiels** : des structures composées de nombreuses fonctions interconnectées, capables d'apprendre des représentations complexes. Le DL est une sous-catégorie du ML.

**Q16. Quelle est la hiérarchie entre IA, ML et DL ?**
→ IA ⊃ ML ⊃ DL. Le Deep Learning est inclus dans le Machine Learning, qui est lui-même inclus dans l'Intelligence Artificielle.

**Q17. Qui a proposé le neurone formel et quand ?**
→ En **1943**, les mathématiciens **Warren McCulloch** et **Walter Pitts** ont proposé le premier modèle mathématique du neurone biologique, appelé **neurone formel**.

---

## SECTION 2 — Du Biologique à l'Artificiel

**Q18. Quels sont les quatre composants clés d'un neurone biologique et leur rôle ?**
→
- **Dendrites** : extensions qui reçoivent les signaux provenant d'autres neurones (= entrées).
- **Synapses** : points de contact entre neurones où les signaux électriques sont transmis.
- **Axone** : prolongement qui conduit le signal vers d'autres neurones (= sortie).
- **Noyau** : joue un rôle central dans le fonctionnement cellulaire et le déclenchement de l'activation neuronale.

**Q19. Combien de synapses possède en moyenne un neurone biologique ? Que se passe-t-il quand le seuil est dépassé ?**
→ En moyenne, un neurone possède environ **10 000 synapses**. Lorsque la somme des signaux reçus dépasse un certain **seuil**, le neurone s'active et génère un signal électrique appelé **potentiel d'action**, qui circule le long de l'axone jusqu'aux synapses.

**Q20. Décrivez le neurone formel de McCulloch et Pitts.**
→ Le neurone formel reçoit un ensemble d'**entrées binaires** $X = [x_1, x_2, \ldots, x_n]^T$ (chaque $x_i \in \{0, 1\}$) et produit une **sortie binaire** $y \in \{0, 1\}$. Chaque entrée $x_i$ est associée à un **poids synaptique** $w_i$. Son fonctionnement se déroule en deux phases.

**Q21. Quelles sont les deux phases du fonctionnement du neurone formel ?**
→
1. **Phase d'agrégation** : calcul de la somme pondérée $S = \sum_{i=1}^{n} x_i w_i$.
2. **Phase d'activation** : application de la fonction seuil (Heaviside) :
$$y = f\left(\sum_{i=1}^n x_i w_i\right) = \begin{cases} 1 & \text{si } \sum_{i=1}^n x_i w_i \geq \theta \\ 0 & \text{sinon} \end{cases}$$

**Q22. Qu'est-ce que la fonction d'activation (ou fonction de transfert) ?**
→ La fonction d'activation détermine la réponse d'un neurone en fonction de sa stimulation. Elle agit comme un seuil : si l'entrée dépasse une certaine valeur, le neurone s'active et transmet un signal. Pour une entrée réelle $x \in \mathbb{R}$, elle produit une sortie $y = f(x)$.

**Q23. Donnez la formule et les propriétés de la fonction Heaviside.**
→
$$y = H(x) = \begin{cases} 0 & \text{si } x < 0 \\ 1 & \text{si } x \geq 0 \end{cases}$$
Propriétés : fonction binaire, saut brusque au seuil, sortie $\in \{0, 1\}$, non dérivable en 0. C'est la fonction d'activation originale du neurone formel.

**Q24. Donnez la formule et les propriétés de la fonction sigmoïde.**
→
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
Propriétés : **continue et dérivable**, sortie dans $]0, 1[$, transition douce en forme de « S ». Avantages : facilite l'optimisation et la descente de gradient ; activation progressive autour de 0.

**Q25. Donnez la formule et les propriétés de la fonction tanh.**
→
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
Propriétés : similaire à la sigmoïde, sortie dans $]-1, +1[$, **fonction impaire** (symétrique par rapport à l'origine), aide à centrer les données.

**Q26. Donnez la formule et les propriétés de la fonction ReLU.**
→
$$f(x) = \max(0, x) = \begin{cases} 0 & \text{si } x < 0 \\ x & \text{si } x \geq 0 \end{cases}$$
Propriétés : la fonction d'activation **la plus utilisée** en Deep Learning moderne. Avantages : sortie nulle pour entrées négatives, proportionnelle pour entrées positives ; facilite l'apprentissage et **limite le problème de disparition du gradient** rencontré avec la sigmoïde.

**Q27. Comment réaliser une porte OU avec un neurone formel ?**
→ Un neurone formel à deux entrées avec des poids $w_1 = w_2 = 1$ et un seuil $\theta = 1$ réalise l'opération **OU** : la sortie est 1 si au moins une entrée vaut 1.

**Q28. Comment réaliser une porte ET avec un neurone formel ?**
→ Un neurone formel à deux entrées avec des poids $w_1 = w_2 = 1$ et un seuil $\theta = 2$ réalise l'opération **ET** : la sortie est 1 uniquement si les deux entrées valent 1.

**Q29. Quelle est la principale limitation du neurone formel de McCulloch et Pitts ?**
→ Le neurone formel ne possède **pas de mécanisme d'apprentissage automatique** : les poids synaptiques doivent être définis **manuellement** pour chaque application. Cela limite fortement son usage pratique.

---

## SECTION 3 — Le Perceptron

**Q30. Qu'est-ce que le perceptron ? Qui l'a inventé et quand ?**
→ Le perceptron est un algorithme d'apprentissage supervisé, inventé par **Frank Rosenblatt en 1957**. C'est l'un des modèles les plus anciens et les plus simples des réseaux de neurones artificiels, utilisé principalement pour la **classification binaire**. C'est le premier algorithme permettant d'apprendre automatiquement les poids d'un neurone artificiel à partir d'exemples.

**Q31. Comment fonctionne le perceptron ?**
→ Le perceptron :
1. Calcule la somme pondérée : $S = \sum_{i=1}^n x_i w_i$
2. Applique une fonction d'activation seuil (Heaviside) :
   - Sortie = 1 si $\sum_{i=1}^n x_i w_i > \Theta$
   - Sortie = 0 sinon

**Q32. Quel est le rôle du biais dans le perceptron ?**
→ Une entrée supplémentaire $x_0 = 1$ est ajoutée, associée à un poids $w_0 = -\Theta$ (où $\Theta$ est le seuil). Cette technique permet d'**incorporer le seuil directement** dans le calcul de la somme pondérée et de **déplacer la frontière de décision**, ce qui est essentiel pour modéliser correctement les données.

**Q33. Quel est le rôle des poids dans le perceptron ?**
→
- Un **poids positif** renforce l'impact de l'entrée correspondante.
- Un **poids négatif** diminue l'impact de l'entrée.
- Un poids **proche de 0** signifie que l'entrée est peu pertinente pour la classification.

**Q34. Quelle est la règle d'apprentissage du perceptron ?**
→ Si le perceptron fait une erreur de classification, il met à jour ses poids selon :
$$w_i^{(t+1)} = w_i^{(t)} + \eta(y - \hat{y}) x_i$$
où $y$ est la vraie étiquette, $\hat{y}$ est la prédiction, et $\eta$ est le taux d'apprentissage (paramètre positif).

**Q35. Qu'est-ce qu'un classifieur linéaire ? Quelle est la frontière de décision du perceptron ?**
→ Le perceptron est un **classifieur linéaire** qui sépare l'espace d'entrée en deux classes. Sa fonction de décision est :
$$\text{signe}(x_1, \ldots, x_n) = \begin{cases} 1 & \text{si } \sum_{i=1}^n w_i x_i + w_0 > 0 \\ -1 & \text{sinon} \end{cases}$$
La frontière entre les deux classes est un **hyperplan** défini par $\sum_{i=1}^n w_i x_i + w_0 = 0$.

**Q36. Quelle est la limite fondamentale du perceptron ?**
→ Le perceptron ne peut apprendre que des problèmes **linéairement séparables**. Des tâches plus complexes (comme XOR) nécessitent des modèles plus puissants (MLP, réseaux profonds).

---

## SECTION 4 — Algorithmes d'Apprentissage

**Q37. Qu'est-ce que la loi de Hebb ?**
→ La loi de Hebb est un modèle biologique d'apprentissage synaptique proposé par le neuropsychologue Donald Hebb :
- Lorsque deux neurones sont **activés simultanément**, leur connexion synaptique est **renforcée**.
- Si les deux neurones ne s'activent **pas en même temps**, leur connexion est **affaiblie ou éliminée**.

**Q38. Donnez la formule de mise à jour de la loi de Hebb.**
→
$$w_i = w_i + \beta(x_i x_j) \quad \Leftrightarrow \quad \Delta w(i,j) = \beta(x_i x_j)$$
où $\beta$ est le **taux d'apprentissage** et $x_i$, $x_j$ sont les activations des neurones d'entrée et de sortie.

**Q39. Donnez le tableau de coactivation de la loi de Hebb.**
→

| $x_i$ | $x_j$ | $\Delta w(i,j) = x_i \cdot x_j$ |
|--------|--------|----------------------------------|
| 0      | 0      | 0                                |
| 0      | 1      | 0                                |
| 1      | 0      | 0                                |
| **1**  | **1**  | **+**                            |

La connexion n'est renforcée que si les **deux neurones sont actifs** simultanément.

**Q40. Décrivez le déroulement général de l'algorithme du perceptron (loi de Hebb).**
→
1. Initialiser aléatoirement les poids $w_i$ et le seuil $\Theta$ (valeurs faibles).
2. Sélectionner un exemple $(X_i, y_i^d)$ dans la base $B$.
3. Calculer la sortie : $\hat{y}_i = \text{signe}(S - \Theta)$ où $S = \sum_k x_k w_k$.
4. Si la sortie est incorrecte ($y_i^d \neq \hat{y}_i$) : mettre à jour chaque poids $w_k = w_k + \beta(x_k y_i^d)$.
5. Répéter jusqu'à ce que tous les exemples soient classés correctement.
6. Retourner les poids $w_i$.

**Q41. Quelle est la limitation de la loi de Hebb ?**
→ La loi de Hebb ne fonctionne **pas toujours**, même lorsque le problème est linéairement séparable. Elle ne converge pas systématiquement vers une solution correcte.

**Q42. Qu'est-ce que la règle delta ? Quel est son principe ?**
→ La règle delta est une méthode d'apprentissage qui ajuste les poids en fonction de l'**erreur de sortie**. Son principe : si l'erreur est grande, la modification des poids doit être importante. Elle tient compte explicitement de l'erreur observée, contrairement à Hebb.

**Q43. Donnez la formule de mise à jour de la règle delta.**
→
$$w_i = w_i + \beta x_i (y_i^d - \hat{y}_i) = w_i + \beta x_i (\text{Err}_i)$$
où $\beta$ est le taux d'apprentissage, $y_i^d$ la sortie désirée, $\hat{y}_i$ la sortie prédite, $\text{Err}_i = y_i^d - \hat{y}_i$ l'erreur observée.

**Q44. Que se passe-t-il selon les différents cas d'erreur dans la règle delta ?**
→
- Si $\text{Err}_i = 0$ (prédiction correcte) → le poids **ne change pas**.
- Si $\text{Err}_i > 0$ (prédiction trop faible) → le poids **augmente**.
- Si $\text{Err}_i < 0$ (prédiction trop forte) → le poids **diminue**.

**Q45. Décrivez les 5 étapes du processus d'apprentissage par règle delta.**
→
1. **Calcul de la sortie prédite** : $\hat{y} = f(w \cdot x + w_0)$
2. **Calcul de l'erreur** : $\text{Err} = y^d - \hat{y}$
3. **Mise à jour des poids** : $w_i \leftarrow w_i + \beta x_i (\text{Err})$
4. **Mise à jour du biais** : $w_0 \leftarrow w_0 + \beta(\text{Err})$
5. **Répétition** pour tous les exemples, puis plusieurs époques, jusqu'à convergence.

**Q46. Qu'est-ce que le risque empirique (fonction d'erreur) ?**
→ Le risque empirique représente la **moyenne des erreurs** commises sur chaque observation étiquetée $(X_i, y_i^d)$ :
$$\text{loss} = E(w) = \frac{1}{N} \sum_{i=1}^N \text{Err}_i(w)$$
Cette quantité mesure la qualité du modèle pour un vecteur de poids $w$ donné. La **fonction d'erreur dépend directement des poids**.

**Q47. Quelle est la fonction d'erreur pour un problème de régression ?**
→ L'erreur quadratique (MSE — Mean Squared Error) :
$$\text{Err}_i(w) = (y_i^d - \hat{y}_i)^2$$

**Q48. Quelle est la fonction d'erreur pour un problème de classification binaire ?**
→ La log-loss (entropie croisée binaire) :
$$\text{Err}_i(w) = -y_i^d \log(\hat{y}_i) - (1 - y_i^d) \log(1 - \hat{y}_i)$$
Utilisée lorsque la sortie représente une probabilité.

**Q49. Quelle est la fonction d'erreur pour un problème de classification multiclasse ?**
→ L'entropie croisée multiclasse ($K > 2$ classes), utilisée avec la fonction softmax :
$$\text{Err}_i(w) = -\sum_{k=1}^K y_k^d \log(\hat{y}_k)$$

**Q50. Qu'est-ce que la convexité d'une fonction d'erreur et pourquoi est-ce important ?**
→
- **Fonction convexe** : le minimum est **unique**. Quel que soit le point de départ, l'algorithme converge vers la même solution optimale.
- **Fonction non convexe** : la fonction peut posséder plusieurs **minima locaux**. Le résultat dépend du point de départ et de la trajectoire.
C'est important car cela détermine si l'apprentissage convergera vers la solution optimale globale.

**Q51. Qu'est-ce que le gradient d'une fonction ? Donnez la définition formelle.**
→ Le gradient d'une fonction $f$ de $n$ variables est le **vecteur de ses dérivées partielles** :
$$\nabla f(x_1, x_2, \ldots, x_n) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$
Ce vecteur indique la **direction de la variation la plus rapide** de la fonction.

**Q52. Quelle est la propriété fondamentale du gradient pour minimiser une fonction ?**
→ Le vecteur gradient, évalué en un point, indique la direction dans laquelle la fonction **augmente le plus rapidement**. Pour **minimiser** la fonction, on se déplace dans la **direction opposée** au gradient.

**Q53. Décrivez l'algorithme de descente de gradient.**
→
1. Choisir une valeur initiale $x_0$ et un taux d'apprentissage $\beta > 0$.
2. Répéter jusqu'à convergence :
   - (a) Calculer la correction : $\Delta x = -\beta \nabla f(x_{t-1})$
   - (b) Mettre à jour : $x_t = x_{t-1} + \Delta x$
La convergence est atteinte quand $x$ n'évolue plus (gradient nul = minimum).

**Q54. Qu'est-ce qu'une epoch ?**
→ Une **epoch** correspond à une **itération complète** sur l'ensemble des exemples d'apprentissage.

**Q55. Quelles sont les trois variantes de la descente de gradient ? Décrivez-les.**
→
1. **Batch (hors-ligne)** : la correction des poids est calculée **après avoir parcouru l'ensemble des exemples**. Très stable mais lent sur de grands jeux de données.
2. **Stochastique (en-ligne)** : la correction est effectuée pour **un seul exemple tiré au hasard**. Plus rapide et mieux tolérant au bruit, mais convergence plus chaotique/irrégulière.
3. **Mini-batch** : la correction est calculée sur **un petit sous-ensemble d'exemples**. Combine les avantages des deux méthodes précédentes.

**Q56. Donnez les formules de mise à jour des poids et du biais dans l'algorithme d'apprentissage complet.**
→
$$\Delta w_i = -\beta \nabla E(w) = -\beta \frac{\partial E}{\partial w_i} = \frac{\beta}{N} X_i^T (y_i^d - \hat{y}_i)$$
$$\Delta w_0 = -\beta \nabla E(w_0) = \frac{\beta}{N} \sum_{i=0}^n (y_i^d - \hat{y}_i)$$

**Q57. Comment détecter que les données ne sont pas linéairement séparables ?**
→ Si, au cours de l'exécution de l'algorithme, on rencontre **deux fois le même vecteur de poids $w$**, cela signifie que les données d'apprentissage **ne sont pas linéairement séparables**.

**Q58. Quelle est la borne maximale d'itérations si les données sont linéairement séparables ?**
→ Si les données sont linéairement séparables, le nombre maximal d'itérations est :
$$(N+1)^2 \cdot 2^{(N+1)\log(N+1)}$$
où $N$ est le nombre d'exemples.

**Q59. Quelles sont les remarques importantes sur le taux d'apprentissage $\beta$ ?**
→
- $\beta$ doit être choisi correctement : **ni trop grand, ni trop petit**.
- Si $\beta$ est **trop grand** : risque d'**oscillations** autour du minimum.
- Si $\beta$ est **trop petit** : nombre **élevé d'itérations** nécessaires.
- En pratique, $\beta$ est souvent **diminué progressivement** au cours des itérations.

**Q60. Quelle est la limite de généralisation du perceptron ?**
→ Le perceptron est **difficilement généralisable à plus de deux classes**. Il ne converge que si les deux classes sont bien séparées.
