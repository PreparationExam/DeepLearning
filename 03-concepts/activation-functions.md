# 🧠 Concepts Fondamentaux : Intelligence Artificielle & Neurones

Ce dossier contient l'architecture conceptuelle du cours, retraçant l'évolution de l'IA depuis les bases biologiques jusqu'aux prémices du Deep Learning.

---

## 1. Cartographie : Du Neurone Biologique au Neurone Formel
[cite_start]Le passage du biologique à l'artificiel repose sur une simplification mathématique des fonctions vitales de la cellule nerveuse[cite: 161, 163].

* [cite_start]**Dendrites $\rightarrow$ Entrées ($x_i$)** : Reçoivent les signaux provenant d'autres neurones[cite: 165].
* [cite_start]**Synapses $\rightarrow$ Poids ($w_i$)** : Représentent l'importance ou la force de la connexion[cite: 166, 201].
* [cite_start]**Noyau/Corps cellulaire $\rightarrow$ Somme pondérée ($\sum$)** : Calcule l'agrégation des signaux reçus[cite: 168, 203, 205].
* [cite_start]**Axone $\rightarrow$ Sortie ($y$)** : Conduit le signal électrique vers d'autres neurones après activation[cite: 167, 193].

> [cite_start]**Principe clé** : Un neurone biologique s'active (potentiel d'action) lorsque la somme des signaux reçus dépasse un certain seuil[cite: 190].

---

## 2. Comparaison des Fonctions d'Activation
[cite_start]La fonction d'activation détermine la réponse du neurone en fonction de sa stimulation[cite: 221].

| Fonction | Équation | Usage & Propriétés | Pourquoi ce choix ? |
| :--- | :--- | :--- | :--- |
| [cite_start]**Heaviside (Seuil)** [cite: 232, 235] | $f(x) = \begin{cases} 1 & \text{si } x \ge 0 \\ 0 & \text{sinon} \end{cases}$ | [cite_start]Perceptron original[cite: 232]. | [cite_start]Simple mais non dérivable, limite l'optimisation moderne[cite: 254]. |
| [cite_start]**Sigmoïde** [cite: 243, 245] | $\sigma(x) = \frac{1}{1+e^{-x}}$ | [cite_start]Sortie entre 0 et 1[cite: 244]. | [cite_start]Continue et dérivable, idéale pour les probabilités[cite: 254]. |
| [cite_start]**Tanh** [cite: 256, 258] | $\tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | [cite_start]Sortie entre -1 et +1[cite: 256]. | [cite_start]Symétrique par rapport à l'origine, aide à centrer les données[cite: 257]. |
| [cite_start]**ReLU** [cite: 264, 266] | $f(x) = \max(0, x)$ | [cite_start]Standard du Deep Learning[cite: 264]. | [cite_start]Simple, rapide et limite la disparition du gradient[cite: 265, 276]. |

---

## 3. Algorithmes d'Apprentissage : Hebb vs Règle Delta

### 🔬 Loi de Hebb (Inspiration Biologique)
* [cite_start]**Concept** : Si deux neurones sont activés simultanément, leur connexion est renforcée[cite: 431].
* [cite_start]**Formule** : $\Delta w(i, j) = \beta(x_i x_j)$[cite: 450].
* [cite_start]**Limite** : Ne fonctionne pas toujours, même pour des problèmes linéairement séparables[cite: 487, 490].

### 📐 Règle Delta (Approche par l'Erreur)
* [cite_start]**Concept** : Ajuste les poids en fonction de l'erreur observée entre la sortie désirée ($y^d$) et la sortie prédite ($\hat{y}$)[cite: 491, 498].
* [cite_start]**Formule** : $w_i = w_i + \beta x_i (y_i^d - \hat{y}_i)$[cite: 528].
* [cite_start]**Optimisation** : Utilise la **descente de gradient** pour minimiser la fonction d'erreur (MSE, Log-loss, etc.)[cite: 566, 570, 602].

---

## 4. Contexte Historique & Évolution
1.  **1943 - Neurone de McCulloch & Pitts** : Premier modèle mathématique. [cite_start]Pas d'apprentissage automatique, poids fixés manuellement[cite: 140, 306].
2.  [cite_start]**1957 - Perceptron de Rosenblatt** : Introduction de la règle d'apprentissage automatique[cite: 308].
3.  [cite_start]**Moderne - Deep Learning** : Passage d'un modèle unique à un réseau de fonctions interconnectées capables de modéliser des représentations complexes[cite: 134, 138].

---

## ⚠️ Pièges à éviter pour l'examen
* [cite_start]**Séparabilité Linéaire** : Un perceptron simple ne peut résoudre un problème que si les classes peuvent être séparées par une ligne droite (hyperplan)[cite: 341, 356, 401].
* [cite_start]**Le rôle du Biais ($w_0$)** : Il permet de déplacer la frontière de décision pour mieux modéliser les données[cite: 345].
* **Le Taux d'Apprentissage ($\beta$)** :
    * [cite_start]Trop grand : Risque d'oscillations autour du minimum[cite: 681].
    * [cite_start]Trop petit : Convergence trop lente (trop d'itérations)[cite: 682].
* [cite_start]**Epoch** : Une epoch est une itération complète sur l'ensemble du jeu de données[cite: 649].

---

## 🎯 Questions de révision type examen
1.  Quelle est la différence majeure entre le neurone de McCulloch-Pitts et le Perceptron ? [cite_start](Réponse : L'apprentissage automatique des poids [cite: 306, 308]).
2.  Pourquoi la fonction ReLU est-elle plus utilisée que la Sigmoïde dans les réseaux profonds ? [cite_start](Réponse : Elle limite la disparition du gradient [cite: 276]).
3.  Que se passe-t-il si les données ne sont pas linéairement séparables ? [cite_start](Réponse : Le perceptron ne peut pas converger vers une erreur nulle [cite: 356, 676]).
4.  Citez une tâche d'apprentissage non supervisé. [cite_start](Réponse : Le clustering ou la réduction de dimension [cite: 73, 83]).
5.  Comment la règle Delta corrige-t-elle un poids si la prédiction est trop faible ? [cite_start](Réponse : Elle augmente le poids [cite: 538]).
# ❓ Quiz Conceptuel Avancé (Architecte Conceptuel)

1.  [cite_start]**Le dilemme de l'interprétabilité** Pourquoi le passage du ML classique au Deep Learning (réseaux de fonctions interconnectées) rend-il le modèle plus difficile à expliquer pour un humain ? [cite: 132, 134, 138]

2.  **Diagnostic du Taux d'Apprentissage ($\beta$)** Si l'erreur ($loss$) de ton modèle oscille violemment sans jamais descendre vers le minimum, est-ce un signe que $\beta$ est trop grand ou trop petit ? [cite_start]Justifie. [cite: 680, 681]

3.  [cite_start]**Logique et Seuils ($\theta$)** Pour transformer un neurone formel à deux entrées ($w_1=1, w_2=1$) en une porte logique **ET (AND)**, quelle est la valeur minimale du seuil $\theta$ ? [cite: 283]

4.  [cite_start]**Alerte sur la Séparabilité** Si au cours de l'apprentissage, tu vois apparaître deux fois exactement le même vecteur de poids $w$, que peux-tu conclure sur ton jeu de données ? [cite: 676]

5.  [cite_start]**Contrainte Mathématique : Heaviside vs Sigmoïde** Pourquoi la fonction de Heaviside est-elle inutilisable pour un algorithme de descente de gradient, contrairement à la Sigmoïde ? [cite: 254, 608]

6.  [cite_start]**Physiologie du Neurone Formel** À quel mécanisme biologique correspond la "Phase d'agrégation" ($S = \sum x_i w_i$) dans le modèle de McCulloch et Pitts ? [cite: 190, 203, 205]

7.  [cite_start]**Limites du Perceptron de Rosenblatt** Pourquoi ce modèle échoue-t-il systématiquement sur des problèmes comme le "OU Exclusif" (XOR) ou la reconnaissance de 10 classes de chiffres ? [cite: 356, 357, 685]

8.  [cite_start]**Signification d'un Poids Nul** Après un long entraînement, si le poids $w_i$ associé à une entrée $x_i$ est proche de 0, quelle est l'influence de cette entrée sur la décision finale ? [cite: 344]

9.  [cite_start]**Choix de l'Algorithme (Big Data)** Pour un dataset de plusieurs millions d'exemples, pourquoi préfère-t-on la descente de gradient **Stochastique** au mode **Batch** ? [cite: 655, 658, 663]

10. [cite_start]**Hebb vs Delta : La philosophie de l'erreur** Quelle est la différence fondamentale entre la loi de Hebb et la règle Delta concernant l'utilisation du résultat prédit pour mettre à jour les poids ? [cite: 430, 491, 492, 498]
