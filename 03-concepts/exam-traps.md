## 3. 🪤 Les Véritables Pièges de l'Examen (Questions de Cours)

Voici les nuances sur lesquelles les étudiants perdent souvent des points :

### A. Le Piège de la Convergence
* **Question type** : "L'algorithme du perceptron finit-il toujours par classer les données ?"
* **Le Piège** : Répondre "Oui".
* **La Vérité** : Il ne converge **que si** les données sont linéairement séparables. S'il y a un chevauchement entre les classes, l'algorithme bouclera à l'infini (oscillation des poids).

### B. Le Piège du Gradient (Direction vs Sens)
* **Question type** : "Dans quelle direction le vecteur gradient pointe-t-il ?"
* **Le Piège** : Répondre "Vers le minimum de l'erreur".
* **La Vérité** : Le gradient pointe vers la direction de la **plus forte augmentation**. Pour minimiser l'erreur, on doit aller dans le sens **opposé** au gradient ($-\nabla$).

### C. Le Piège du Biais ($w_0$)
* **Question type** : "À quoi sert le biais dans un perceptron ?"
* **Le Piège** : Répondre "C'est juste un poids de plus".
* **La Vérité** : Sans biais, la frontière de décision passe obligatoirement par l'origine $(0,0)$. Le biais permet de déplacer la ligne/l'hyperplan dans l'espace pour s'adapter aux données décentrées.

### D. Hebb vs Delta
* **Question type** : "La règle de Hebb est-elle suffisante pour l'apprentissage supervisé ?"
* **Le Piège** : Répondre "Oui, car elle renforce les liens".
* **La Vérité** : Non. Hebb renforce la co-activation, mais la **Règle Delta** est supérieure car elle base la correction sur l'**erreur réelle** ($y^d - \hat{y}$).

### E. Initialisation des Poids
* **Question type** : "Peut-on initialiser tous les poids à zéro ?"
* **Le Piège** : Répondre "Oui, le gradient les corrigera".
* **La Vérité** : Si tous les poids sont identiques (ex: tous à zéro), tous les neurones d'une même couche feront exactement la même mise à jour (problème de symétrie). L'apprentissage sera bloqué ou inefficace.