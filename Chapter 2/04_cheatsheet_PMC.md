# 🗒️ CHEAT SHEET — PMC (Tout sur une page)

```
╔══════════════════════════════════════════════════════════════════════╗
║                    PERCEPTRON MULTI-COUCHES                          ║
╠══════════════════════════════════════════════════════════════════════╣
║ ARCHITECTURE                                                         ║
║   [X] → [Cachée(s)] → [Sortie]    Fully connected entre couches     ║
║   Entrée: n = nb features          Cachée: expérimentation           ║
║   Sortie: 1 (régression/binaire) | K (multiclasse)                   ║
║   ≥ 2 couches cachées = réseau PROFOND                               ║
╠══════════════════════════════════════════════════════════════════════╣
║ ACTIVATION DE SORTIE                                                 ║
║   Régression     → Linéaire (rien)                                   ║
║   Classif binaire → Sigmoïde σ(x) = 1/(1+e^{-x})   seuil: 0.5      ║
║   Classif K-classes → Softmax    seuil: argmax                       ║
║   Couches cachées → même activation (souvent sigmoïde)              ║
╠══════════════════════════════════════════════════════════════════════╣
║ FORMULES ESSENTIELLES                                                ║
║                                                                      ║
║  Sortie neurone j:   ŷⱼ = Σ w(i,j)·ŷᵢ + w(0,j)                    ║
║  Sigmoïde:           σ(y) = 1/(1+e^{-y})                            ║
║  Dérivée sigmoïde:   σ'(x) = ŷⱼ(1 - ŷⱼ)                           ║
║                                                                      ║
║  ▶ Gradient local SORTIE:                                            ║
║    δⱼ = ŷⱼ(1-ŷⱼ)(yⱼᵈ - ŷⱼ)                                        ║
║                                                                      ║
║  ▶ Gradient local CACHÉ:                                             ║
║    δⱼ^[k-1] = ŷⱼ(1-ŷⱼ) Σ w(j,r)δᵣ^[k]                            ║
║                          r∈dest(j)                                   ║
║                                                                      ║
║  ▶ Mise à jour poids:  ΔW^[k] = (β/m) δ^[k] (Ŷ^[k-1])^T           ║
║  ▶ Mise à jour biais:  Δb^[k] = (β/m) Σ δⱼ^[k]                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ ALGORITHME (ordre à retenir)                                         ║
║   1. Normaliser: xᵢ ∈ [-1,1]  |  yᵈ ∈ {0.05, 0.95}                ║
║   2. Init poids: w(i,j) ∈ [-0.5, 0.5] aléatoire                     ║
║   3. Boucle:                                                         ║
║      → Tirer exemple (X,yᵈ) au hasard                               ║
║      → Propagation AVANT (couche par couche → sortie)               ║
║      → Calcul δ SORTIE puis δ CACHÉS (sortie → entrée)              ║
║      → Mise à jour: w ← w + Δw                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║ CHOIX DU RÉSEAU                                                      ║
║  Tâche                  | Sortie | Activation                        ║
║  Régression simple      |   1    | Aucune                            ║
║  Régression multi (p)   |   p    | Aucune                            ║
║  Classification binaire |   1    | Sigmoïde (seuil 0.5)             ║
║  Classification K-class |   K    | Softmax (argmax)                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ PROBLÈMES ET SOLUTIONS                                               ║
║  Saturation    → normaliser données [-1,1]                           ║
║  Surapprentissage → L1/L2 régularisation, dropout, + données        ║
║  β trop grand  → divergence / oscillations                           ║
║  β trop petit  → convergence très lente                              ║
║  Init à zéro   → INTERDIT (symétrie: tous les neurones identiques)  ║
╠══════════════════════════════════════════════════════════════════════╣
║ RÉGULARISATION                                                       ║
║  L1 (Lasso)  : Ω = λΣ|wᵢ|     → poids = 0 (sparse)                ║
║  L2 (Ridge)  : Ω = λΣwᵢ²      → poids petits                       ║
║  Dropout     : désactive x% neurones aléatoires pendant train        ║
║  Data aug.   : flip, rotate, crop, scale (images)                   ║
║  λ = 0 → pas de régul.  |  λ grand → régul. forte                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ XOR (EXEMPLE CLASSIQUE)                                              ║
║  Non linéairement séparable → perceptron simple échoue              ║
║  PMC avec 1 couche cachée (2 neurones) → résolu                     ║
║  x₁ XOR x₂ = (x₁ OR x₂) AND (NOT(x₁) OR NOT(x₂))                 ║
║                                                                      ║
║  Preuve impossibilité linéaire:                                      ║
║  (0,0)→0: w₀=0  |  (1,0)→1: w₁+w₀=1  |  (0,1)→1: w₂+w₀=1        ║
║  → w₁+w₂+w₀=2   MAIS   (1,1)→0: w₁+w₂+w₀=0   CONTRADICTION ✗     ║
╠══════════════════════════════════════════════════════════════════════╣
║ ONE-HOT ENCODING                                                     ║
║  K classes → vecteur de taille K, 1 à la position de la classe     ║
║  Ex: 3 classes → rouge=[1,0,0]  vert=[0,1,0]  bleu=[0,0,1]         ║
║  Classe prédite = argmax(ŷⱼ)                                         ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Dérivées Utiles

| Fonction | Dérivée |
|----------|---------|
| σ(x) = 1/(1+e⁻ˣ) | σ(x)·(1-σ(x)) = ŷ(1-ŷ) |
| ReLU(x) = max(0,x) | 0 si x<0, 1 si x>0 |
| tanh(x) | 1 - tanh²(x) |

## Symboles à connaître

| Symbole | Signification |
|---------|--------------|
| β | Taux d'apprentissage |
| δⱼ | Gradient local du neurone j |
| w(i,j) | Poids de la connexion i→j |
| w(0,j) | Biais du neurone j |
| ŷⱼ | Sortie prédite du neurone j |
| yⱼᵈ | Valeur désirée (label) |
| ⊙ | Produit de Hadamard (élément par élément) |
| m | Nombre d'exemples d'entraînement |
| R | Nombre de neurones de la couche précédente |
| λ | Paramètre de régularisation |
