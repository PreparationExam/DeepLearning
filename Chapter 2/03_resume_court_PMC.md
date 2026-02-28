# ⚡ Résumé Court — PMC (L'essentiel uniquement)

---

## 1. Pourquoi le PMC existe
- Perceptron simple = **linéaire uniquement** → échoue sur XOR
- Minsky & Papert (1969) : preuve formelle des limites
- Solution : **couches successives** → transformations non linéaires
- **Théorème approx. universelle** : un PMC peut approximer n'importe quelle fonction

---

## 2. Architecture
```
[Entrée] → [Cachée 1] → [Cachée 2] → ... → [Sortie]
```
- **Entrée** : nb neurones = nb features
- **Cachée(s)** : choix empirique, ≥2 couches cachées = réseau **profond**
- **Sortie** : dépend de la tâche

| Tâche | Neurones sortie | Activation |
|-------|-----------------|-----------|
| Régression | 1 (ou p) | Linéaire |
| Classif. binaire | 1 | **Sigmoïde** |
| Classif. multiclasse | K | **Softmax** |

- Neurones même couche : **non connectés**
- Neurones couches adjacentes : **tous connectés (fully connected)**

---

## 3. Formules clés

**Sortie régression :**
$$\hat{y}_j = \sum_{i=1}^{R} w(i,j)\hat{y}_i + w(0,j)$$

**Sortie classif. binaire :** appliquer σ à la somme pondérée → si > 0.5 : C₁

**Gradient local (sortie) :**
$$\delta_j = \hat{y}_j(1 - \hat{y}_j)(y_j^d - \hat{y}_j)$$

**Gradient local (caché) :**
$$\delta_j^{[k-1]} = \hat{y}_j(1-\hat{y}_j) \sum_{r} w(j,r)\delta_r^{[k]}$$

**Mise à jour poids :**
$$\Delta W = \frac{\beta}{m} \delta \cdot (\hat{Y}^{prev})^T$$

---

## 4. Algorithme d'apprentissage (5 étapes)
1. Normaliser : xᵢ ∈ [-1, 1], yᵈ ∈ **{0.05, 0.95}**
2. Initialiser : w(i,j) ∈ **[-0.5, 0.5]** aléatoirement
3. Choisir aléatoirement un exemple
4. Propagation **avant** → calcul des sorties
5. Propagation **arrière** → calcul des δ + mise à jour des poids

---

## 5. Problèmes pratiques importants
- **Saturation** : neurone avec sortie ≈ 0 ou 1 → gradient ≈ 0 → apprentissage bloqué → **NORMALISER les données**
- **Valeurs désirées** : utiliser 0.05/0.95 au lieu de 0/1 (évite saturation)
- **β (learning rate)** : empirique, entre 0 et 1. Grand β = rapide mais instable. Petit β = lent mais stable.
- **Init. poids** : aléatoire [-0.5, 0.5] → PMC est **stochastique**

---

## 6. Surapprentissage
- Underfitting : erreur train élevée → modèle trop simple
- **Overfitting** : erreur train faible + erreur test élevée → mémorise

**Solutions (régularisation) :**
- **L1 (Lasso)** : Ω = λΣ|wᵢ| → poids → 0 (sélection)
- **L2 (Ridge)** : Ω = λΣwᵢ² → poids petits
- **Dropout** : désactive x% neurones aléatoirement pendant l'entraînement
- **Data augmentation** : flip, rotation, crop, etc.
