# CHEAT SHEET — Chapitre 1 : Réseaux de Neurones

---

## HIÉRARCHIE & TYPES D'APPRENTISSAGE

```
┌─────────────────── IA ───────────────────┐
│  ┌──────────────── ML ────────────────┐  │
│  │  ┌──── DL ────┐  SVM, kNN,        │  │
│  │  │Réseaux de  │  Régression,      │  │
│  │  │neurones    │  Arbres...        │  │
│  │  └────────────┘                   │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘

Supervisé   : B = (Xᵢ, yᵢᵈ)₁≤ᵢ≤ₙ   ← données ÉTIQUETÉES
Non supervisé : B = (Xᵢ)₁≤ᵢ≤ₙ      ← données NON étiquetées
Classification : yᵢᵈ ∈ {1,...,C}
Régression    : yᵢᵈ ∈ ℝ
```

---

## NEURONE FORMEL (McCulloch-Pitts, 1943)

```
x₁ ──[w₁]──┐
x₂ ──[w₂]──┤──[Σ]──[f]──→ y
...         │
xₙ ──[wₙ]──┘

S = Σ xᵢwᵢ    y = 1 si S ≥ θ, sinon 0
```

**Portes logiques** (w₁=w₂=1) :
```
θ=1 → OU  |  θ=2 → ET
```

---

## FONCTIONS D'ACTIVATION

```
HEAVISIDE          SIGMOÏDE             TANH              ReLU
y                  y                    y                  y
1 ┤──────           1 ┤· · · · ·         1 ┤· · · · ·       /
  │                  │      /              │     /          /
0 ┼──/──── x        ½┤    /             0 ┼───/────── x   /
  0                  │  /               -1┤· ·          ──┼──── x
                     0                    0             0

H(x)=0 si x<0     σ(x)=1/(1+e⁻ˣ)    tanh(x)=(eˣ-e⁻ˣ)  f(x)=max(0,x)
      =1 si x≥0   sortie: ]0,1[      /(eˣ+e⁻ˣ)         sortie: [0,+∞[
                                      sortie: ]-1,1[
```

---

## PERCEPTRON (Rosenblatt, 1957)

```
x₀=1 ──[w₀]──┐         Frontière : Σwᵢxᵢ + w₀ = 0
x₁   ──[w₁]──┤──[Σ]──[f]──→ ŷ
...           │              signe = +1 si Σwᵢxᵢ+w₀ > 0
xₙ   ──[wₙ]──┘                      -1 sinon
```
Biais : x₀=1, w₀=-Θ | **Limite** : linéairement séparable UNIQUEMENT

---

## LOI DE HEBB

$$\Delta w(i,j) = \beta \cdot x_i \cdot x_j$$

```
xᵢ | xⱼ | Δw
 0 |  0 |  0
 0 |  1 |  0
 1 |  0 |  0
 1 |  1 |  +     ← SEUL CAS où la connexion se renforce
```

**Algo** : Init w=0 → Pour chaque ex : ŷ=signe(S-Θ) → Si erreur : wₖ=wₖ+β(xₖyᵈ)

---

## RÈGLE DELTA

$$w_i \leftarrow w_i + \beta x_i (y^d - \hat{y}) \qquad \text{Err} = y^d - \hat{y}$$
$$w_0 \leftarrow w_0 + \beta (y^d - \hat{y})$$

```
Err=0 → pas de changement
Err>0 → w augmente
Err<0 → w diminue
```

**Modèle** : $\hat{y} = f\left(\sum_{i=1}^n x_i w_i + w_0\right)$

---

## FONCTIONS DE PERTE

```
Régression (MSE)      : Errᵢ(w) = (yᵈ - ŷ)²
Classif. binaire      : Errᵢ(w) = -yᵈlog(ŷ) - (1-yᵈ)log(1-ŷ)
Multiclasse           : Errᵢ(w) = -Σₖ yₖᵈ log(ŷₖ)

Risque empirique : E(w) = (1/N) Σᵢ Errᵢ(w)
```

**Exemple MSE** : Yᵈ=(1,0,0,0), Ŷ=(0.8,0.2,0.1,0.7)
→ Loss = ¼[(1-0.8)²+(0-0.2)²+(0-0.1)²+(0-0.7)²] = 0.145

---

## GRADIENT & DESCENTE

$$\nabla f(x_1,...,x_n) = \left(\frac{\partial f}{\partial x_1},...,\frac{\partial f}{\partial x_n}\right)$$

**Propriété** : ∇f pointe vers la **montée** → pour descendre : $-\nabla f$

$$\boxed{x_t = x_{t-1} - \beta \nabla f(x_{t-1})}$$

```
Ex: f(x)=(x+1)²-2,  ∇f=2(x+1),  x₀=-4, β=0.1
  x₁ = -4 + 0.1×6 = -3.4
  x₂ = -3.4 + 0.1×4.8 = -2.92
  ... → converge vers x*=-1
```

---

## 3 VARIANTES DESCENTE DE GRADIENT

```
┌─────────────┬─────────────────────────────────┬───────────────────┐
│ Variante    │ Correction calculée sur          │ Caractéristiques  │
├─────────────┼─────────────────────────────────┼───────────────────┤
│ Batch       │ TOUS les exemples                │ Stable, lent      │
│ Stochastique│ 1 exemple aléatoire              │ Rapide, irrégulier│
│ Mini-batch  │ Petit sous-ensemble              │ Compromis ✓       │
└─────────────┴─────────────────────────────────┴───────────────────┘
```

---

## CONVERGENCE & ALGO COMPLET

```
Δwᵢ = (β/N) · Xᵢᵀ · (yᵈ - ŷ)
Δw₀ = (β/N) · Σ(yᵈ - ŷ)

Convergence :
  Convexe     → minimum UNIQUE, convergence garantie
  Non-convexe → minima locaux, dépend du départ
  
  Même w rencontré 2x → NON séparable
  Borne max : (N+1)² · 2^((N+1)log(N+1))
  
β trop grand → oscillations
β trop petit → lent
β optimal   → diminué progressivement
```

---

## RÉSUMÉ DATES CLÉS

| Année | Auteur(s) | Contribution |
|-------|-----------|--------------|
| 1943 | McCulloch & Pitts | Neurone formel |
| 1957 | Frank Rosenblatt | Perceptron |
| ~1949 | Donald Hebb | Loi de Hebb |
