

## 2. Les Fonctions d'Activation : Le Levier de Décision
La fonction d'activation agit comme un seuil : elle détermine si le neurone doit "s'allumer" ou non.

| Fonction | Équation | Propriétés & Usage |
| :--- | :--- | :--- |
| **Heaviside** | $y=1$ si $x \ge \theta$ | Utilisation historique. Sortie binaire (0 ou 1). Non dérivable. |
| **Sigmoïde** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Transition entre 0 et 1. Pour prédire des probabilités. |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Symétrique (-1 à +1). Aide à centrer les données autour de zéro. |
| **ReLU** | $\max(0, x)$ | Standard actuel. Très rapide et évite la disparition du gradient en Deep Learning. |

## 🎯 Quiz de Révision Complet (20 Questions)

### Partie 1 : Fondamentaux
1.  Le neurone de McCulloch-Pitts peut-il apprendre seul ? *(Non, poids manuels)*.
2.  Quelle partie du neurone biologique est simulée par les poids synaptiques ? *(La synapse)*.
3.  Comment s'appelle le premier algorithme d'apprentissage de 1957 ? *(Le Perceptron)*.
4.  Citez deux tâches d'apprentissage non supervisé. *(Clustering, réduction de dimension)*.
5.  Que signifie "données étiquetées" ? *(On connaît la réponse correcte pour chaque exemple)*.
7.  Pourquoi ReLU est-elle préférée à la Sigmoïde en Deep Learning ? *(Évite la disparition du gradient)*.
8.  Qu'est-ce qu'une "Epoch" ? *(Un passage complet sur tout le dataset)*.
9.  Quelle est la différence entre classification et régression ? *(Catégories vs Valeurs continues)*.
10. Que mesure la "Fonction Coût" ? *(L'erreur entre la prédiction et la réalité)*.

### Partie 2 : Analyse et Intuition
11. Pourquoi le Deep Learning est-il moins interprétable que le ML classique ? *(Réseau complexe de fonctions interconnectées)*.
12. Si ton erreur oscille violemment, que fais-tu de ton taux d'apprentissage $\beta$ ? *(Le diminuer)*.
13. Porte **ET (AND)** avec $w_1=1, w_2=1$ : quel est le seuil $\theta$ ? *(Seuil $\theta = 2$)*.
14. Si un poids $w$ se répète deux fois durant l'entraînement, conclusion ? *(Données non linéairement séparables)*.
15. Pourquoi Heaviside est-elle impossible en descente de gradient ? *(Non dérivable)*.
16. À quoi sert physiquement la somme pondérée ? *(Intégration des signaux entrants)*.
17. Pourquoi le perceptron échoue sur le XOR ? *(Problème non linéairement séparable)*.
18. Que signifie un poids $w_i \approx 0$ après entraînement ? *(Entrée non pertinente)*.
19. Pourquoi SGD (Stochastique) pour les Big Data ? *(Rapidité et gestion de la mémoire)*.
20. Différence philosophique Hebb vs Delta ? *(Hebb = co-activation / Delta = correction d'erreur)*.