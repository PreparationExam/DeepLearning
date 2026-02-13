## 3. L'Apprentissage : Hebb vs Règle Delta

### 🔬 La Loi de Hebb
Si deux neurones s'activent ensemble, leur lien se renforce. 

 $\Delta w = \beta(x_i x_j)$. 
 
 Elle est limitée car elle ne prend pas en compte l'erreur finale.

### 📐 La Règle Delta & Descente de Gradient
On compare la sortie prédite ($\hat{y}$) à la vérité ($y^d$). On utilise ensuite le **gradient** pour descendre la pente de l'erreur jusqu'à trouver le minimum.
* **Batch** : Mise à jour après avoir vu TOUTES les données. Stable mais lent.
* **Stochastique** : Mise à jour après CHAQUE exemple. Rapide mais chaotique.
* **Mini-batch** : Compromis optimal utilisé en pratique.
