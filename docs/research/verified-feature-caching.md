# Verified feature caching

Notes de veille sur les techniques de cache à vérification pour modèles
itératifs (diffusion image / vidéo / audio, dLLMs). Cette page documente
le pattern et son applicabilité à `mlx-arsenal` — elle n'engage pas un
ADR ni une roadmap.

## Le pattern : forecast-then-verify

Deux familles coexistent dans la littérature 2024-2025 :

### Cache-then-reuse (statu quo de `mlx-arsenal`)

Une heuristique step-à-step (typiquement relative-L1 sur les inputs ou
hidden states) décide si le step courant peut ré-utiliser le résultat du
step précédent — puis applique la décision **sans vérifier la qualité
réelle**.

Implémentations dans la lib : `TeaCacheController`,
`PerLayerAttentionCache`, `PerHeadAttentionCache`,
`WindowResidualController.adaptive`. Toutes utilisent relative-L1.

Limites connues : dérive silencieuse quand l'hypothèse de smoothness
casse (transition de contenu, prompt inhabituel), plafond de speedup
autour de 3-4× avant collapse qualité.

### Forecast-then-verify

Trois étapes :

1. **Draft** — extrapolation paramétrique-free des features
   (typiquement expansion de Taylor sur l'axe itératif via différences
   finies d'ancres précédentes). Coût quasi nul : algèbre linéaire sur
   tenseurs déjà en mémoire.
2. **Verify** — forward partiel sur **une seule couche** (≈1.7-3.5% du
   coût d'un forward complet) pour mesurer l'erreur relative L2 entre
   feature draftée et feature réelle :
   ```
   e_k = ‖F_pred − F_real‖²₂ / (‖F_real‖²₂ + ε)
   ```
3. **Accept/reject** — seuil adaptatif sur l'axe itératif
   (`τ_t = τ₀ · β^((T-t)/T)` chez SpeCa).

Papiers de référence :

- **SpeCa** — *Accelerating Diffusion Transformers with Speculative
  Feature Caching* (Zou et al., sept. 2025). Couche-cible
  empirique (27e sur DiT image), 6-7× speedup avec FID stable.
- **TaylorSeer** — *Forecasting Features rather than Caching Them*
  (mars 2025). Variante sans vérification ; s'effondre à 17.5% de
  dégradation au speedup où SpeCa tient.
- **Spiffy / SSD** (sept-oct. 2025) — speculative decoding pour
  diffusion LLMs (token masking). Même pattern, axe différent.

## Caveat : lossy borné, pas lossless

Contrairement au speculative decoding LLM (lossless par construction
via rejection sampling), SpeCa est **lossy mais borné** : convergence en
variation totale conditionnée à un choix correct de schedule de seuil
(cf. SpeCa Appendix G).

Pour la reproductibilité bit-exact, ne pas utiliser. Pour génération
créative, le compromis est documenté et acceptable.

## Applicabilité à `mlx-arsenal`

### Ce qui est *primitive* (extractible vers la lib)

- L'extrapolation Taylor-via-différences-finies (mais c'est ~20 lignes
  d'algèbre — trop fin pour mériter son propre module).
- L'orchestration draft → verify → accept/reject sur un axe itératif
  abstrait.
- La schedule de seuil géométrique.

### Ce qui reste *caller-side* (per-model)

- **Sélection de la couche-cible de vérification** : pure ablation
  empirique. SpeCa identifie la couche 27 sur leur DiT image ; chaque
  port doit refaire l'ablation contre sa propre métrique de qualité
  (FID, ImageReward, VBench, perplexité…).
- **Calibration de τ₀ et β** : dépend du modèle, du schedule de
  sampling, du domaine.
- **Choix de l'ordre Taylor** : SpeCa utilise typiquement 1-3 selon le
  régime.
- **Définition de la « feature » trackée** : architecture-dépendant.

### Décision actuelle

Un seul contrôleur opinionated, defaults SpeCa, exposé dans
`mlx_arsenal.diffusion.verified_cache`. Pas de batch de primitives
extraites ni d'ADR — la valeur unique vs `TeaCache` est trop fine pour
justifier une roadmap multi-PR, et les vrais points durs (ablation
couche-cible, calibration seuils) sont per-modèle de toute façon.

À ré-évaluer si 2+ ports adoptent le pattern et révèlent des points
durs réutilisables.

## Directions adjacentes notées mais non couvertes ici

- **Parallel sampling via Picard-Lindelöf iteration** (Shih et al. 2023,
  *Accelerating Parallel Sampling* 2024) — résout l'ODE de diffusion
  en parallèle sur plusieurs timesteps. Coût mémoire significatif ;
  intéressant sur Mac Studio M3 Ultra (mémoire unifiée généreuse),
  prohibitif sur M-series modeste.
- **Accelerated Diffusion via Speculative Sampling** (jan-juil. 2025) —
  exploite la connexion speculative sampling ↔ reflection maximal
  coupling pour samplers stochastiques. *Lossless* au sens strict,
  pertinent pour eval/repro.
- **Parallel Sampling via Autospeculation** (nov. 2025) — résultat
  théorique : speedup O(n) → O(√n) en haute précision.

Ces directions n'ont pas (encore) d'implémentation MLX et leur ROI
mid-level n'est pas évident — à creuser indépendamment.

## Références

- SpeCa : <https://arxiv.org/abs/2509.11628>
- TaylorSeer : <https://arxiv.org/abs/2503.06923>
- DiTFastAttn : <https://arxiv.org/abs/2406.08552>
- Repo de référence pour SpeCa : <https://github.com/Shenyi-Z/Cache4Diffusion>
- TeaCache : <https://github.com/ali-vilab/TeaCache>
