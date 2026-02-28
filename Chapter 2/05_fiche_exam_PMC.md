# 🎯 FICHE EXAM — PMC (Ce qui va tomber)

> Basé sur les points les plus insistés dans le cours. Maîtrise ça et tu passes.

---

## 🔴 ULTRA-PRIORITAIRE (tombera presque à coup sûr)

### 1. La preuve XOR (à faire de mémoire)
> "Montrez qu'il est impossible de représenter XOR avec un modèle linéaire."

Supposons XOR(x₁, x₂) = w₁x₁ + w₂x₂ + w₀

- (0,0) → **w₀ = 0**
- (1,0) → w₁ + w₀ = 1 → **w₁ = 1**
- (0,1) → w₂ + w₀ = 1 → **w₂ = 1**
- Donc : w₁ + w₂ + w₀ = **2**
- (1,1) → impose : w₁ + w₂ + w₀ = **0**
- **CONTRADICTION** → impossible ✗

---

### 2. Le gradient local — formules complètes

**Neurone de SORTIE :**
$$\delta_j = \hat{y}_j(1 - \hat{y}_j)(y_j^d - \hat{y}_j)$$

**Neurone CACHÉ :**
$$\delta_j^{[k-1]} = \hat{y}_j^{[k-1]}(1-\hat{y}_j^{[k-1]}) \sum_{r \in \text{dest}(j)} w^{[k]}(j,r)\delta_r^{[k]}$$

**Mise à jour poids :**
$$\Delta W^{[k]} = \frac{\beta}{m} \delta^{[k]} (\hat{Y}^{[k-1]})^T$$

---

### 3. Identification du nombre de neurones en sortie

Donné : "tâche de classification de 5 classes à partir d'images 28×28 pixels"
→ Entrée : 784 neurones, Sortie : **5 neurones** (softmax), one-hot

---

### 4. Le phénomène de saturation
- **Cause** : grandes valeurs d'entrée → sigmoïde ≈ 0 ou 1 → gradient ≈ 0
- **Conséquence** : apprentissage bloqué
- **Solution** : normaliser xᵢ ∈ [-1, 1] + transformer yᵈ ∈ {0.05, 0.95}

---

## 🟠 TRÈS PROBABLE

### 5. Exercice de propagation avant
Savoir calculer :
1. Sⱼ = Σ w(i,j)·xᵢ + w(0,j)  ← somme pondérée
2. ŷⱼ = σ(Sⱼ) = 1/(1+e^{-Sⱼ})  ← application sigmoïde

### 6. Exercice de rétropropagation
Savoir calculer δ des neurones de sortie puis des neurones cachés, puis les ΔW.

### 7. Sur-apprentissage vs sous-apprentissage
- Overfitting : train ↓ / test ↑ (écart)
- Underfitting : train ↑ (erreur haute)
- Régularisation : L1, L2, Dropout

### 8. Régularisation L1 vs L2
| | L1 (Lasso) | L2 (Ridge) |
|-|-----------|-----------|
| Formule | λΣ\|wᵢ\| | λΣwᵢ² |
| Effet | Poids → 0 (sparse) | Poids petits |
| Utilité | Sélection features | Réduction magnitude |

---

## 🟡 PROBABLE

### 9. Définitions clés à rédiger
- **PMC** : réseau de neurones avec ≥1 couche cachée permettant de traiter des problèmes non linéaires
- **Backpropagation** : algorithme calculant les gradients de sortie vers l'entrée pour mettre à jour les poids
- **Dropout** : désactivation aléatoire de x% neurones pendant l'entraînement pour éviter l'overfitting
- **One-hot** : encodage d'une classe par vecteur binaire où 1 = classe prédite
- **Théorème d'approximation universelle** : un PMC peut approximer n'importe quelle fonction

### 10. Architecture selon la tâche
> "Vous voulez prédire le prix d'une maison (régression). Décrivez la couche de sortie."
→ **1 neurone**, **pas de fonction d'activation** (sortie linéaire)

> "Vous classifiez des images dans 10 catégories. Décrivez la couche de sortie."
→ **10 neurones**, **softmax**, classe = argmax(ŷ)

---

## 🟢 BONUS (si tu veux tout écraser)

### 11. Initialisation des poids
- Distribution uniforme **[-0.5, 0.5]**
- Aléatoire (jamais tous à zéro → symétrie → tous neurones identiques)
- PMC = algorithme **stochastique** → résultats légèrement différents à chaque run

### 12. Taux d'apprentissage β
- Trop grand : oscillations, divergence
- Trop petit : convergence très lente
- Stratégie : diminution progressive OU alternance

### 13. Data augmentation
Pour images : flip H/V, rotation, crop, zoom → enrichit le jeu sans nouvelles données réelles

---

## ⚠️ Erreurs classiques à éviter

- Oublier que les neurones d'**une même couche** ne sont pas connectés
- Confondre **sigmoïde** (binaire) et **softmax** (multiclasse)
- Oublier le **biais** w(0,j) dans les calculs
- Ne pas normaliser les données avant de faire tourner le PMC
- Confondre **propagation avant** (calcul des sorties) et **rétropropagation** (calcul des gradients)
- Mettre yᵈ = {0, 1} au lieu de **{0.05, 0.95}** → saturation!
