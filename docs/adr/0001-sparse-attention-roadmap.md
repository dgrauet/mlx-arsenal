# ADR-0001 : Sparse-attention roadmap for video DiTs

- **Statut** : accepted
- **Date** : 2026-05-15
- **Stacks concernées** : `python` / `mlx_arsenal.attention`, `mlx_arsenal.diffusion`

## Contexte

Les diffusion transformers vidéo (LTX-Video, CogVideoX, Wan2,
Hunyuan-Video, etc.) passent la majorité de leur compute dans
l'auto-attention 3D sur une séquence spatiotemporelle de longueur
`S = T·H·W`. La matrice d'attention complète est `S²`, mais la
littérature récente (DiTFastAttn, Sparse VideoGen, Sliding Tile
Attention, Radial Attention, SVG2) montre que la plupart des heads se
comportent comme des heads *spatial-locality* ou *temporal-locality*, et
que des masques structurés restreints à ces patterns récupèrent une
qualité quasi-complète à une fraction du compute.

`mlx-arsenal` doit fournir la **couche mid-level** : les primitives
réutilisables qui permettent de porter ces techniques dans un DiT MLX
sans réimplémenter à chaque port. La couche kernel (Metal block-sparse)
reste hors scope — c'est un projet en soi.

## Décision

Ship six étapes complémentaires sur `attention/` et `diffusion/`,
mergées comme PRs séparées (#27 à #32) sur `main` :

| Étape | Module | Symboles | PR |
|---|---|---|---|
| 1 | `attention.video_masks` | `spatial_only_mask`, `temporal_only_mask`, `sliding_tile_block_mask`, `sliding_tile_centered_mask`, `radial_box_mask`, `radial_gaussian_mask` | #27 |
| 2 | `attention.profile` | `Kind`, `classify`, `classify_heads_from_qk`, `classify_heads_from_probs` | #28 |
| 3 | `diffusion.attention_cache` | `PerLayerAttentionCache`, `PerHeadAttentionCache`, `splice_heads` | #29 |
| 4 | `diffusion.cfg_skip` | `cfg_head_similarity`, `cfg_skip_mask`, `CFGSimilarityProfiler`, `CFGSkipController` | #30 |
| 5 | `diffusion.window_residual` | `WindowResidualController` (3 modes) | #31 |
| 6 | `attention.permute` | `block_contiguous_permutation`, `invert_permutation` | #32 |

### Conventions partagées

- **Token order :** T-major flattening (`[t0(h0w0..hHwW), t1(...), ...]`).
  Aligné avec LTX, CogVideoX et `mlx_arsenal.spatial.patchify`.
- **Mask format :** float additif `0.0` / `-inf`, shape `(1, 1, S, S)`.
  Broadcast natif vers `mx.fast.scaled_dot_product_attention`.
- **Métrique de similarité step-à-step :** relative-L1
  (`mean(|x - prev|) / mean(|prev|)`) pour tous les contrôleurs
  step-aware (`TeaCacheController`, `PerLayerAttentionCache`,
  `PerHeadAttentionCache`, `WindowResidualController.adaptive`).
  Cohérence d'API → un seuil calibré pour TeaCache se transpose
  directement.
- **Métrique cond/uncond :** `cosine` par défaut (littérature ASC),
  `relative_l1` configurable.
- **Boundary policy :** step 0 et `num_steps - 1` forcent toujours le
  refresh / recompute, dans tous les contrôleurs step-aware.
- **Validation :** `ValueError` pour arguments invalides,
  `RuntimeError` pour invariant violations (état non initialisé).
  Aucun `assert` dans `src/` (Python `-O` les strip).

### Découpage en couches

Chaque étape ne dépend que des primitives MLX et (au plus) d'une étape
antérieure :

- Étapes 4–5 ré-utilisent `splice_heads` (étape 3) pour le splice
  cond/uncond et la composition window+residual.
- Étape 2 (profiler) consomme des Q/K mais ne dépend d'aucune autre.
- Étape 6 (permutation) est totalement autonome.

Cette indépendance permet à un caller de ne consommer que ce dont il a
besoin (par exemple WA-RS sans ASC, ou ASC sans TeaCache).

## Conséquences

- 22 nouveaux symboles publics, +104 tests, 364 tests verts en CI.
- Les ports MLX (LTX-2, CogVideoX, Wan2, etc.) peuvent désormais
  composer ces primitives plutôt que les réimplémenter.
- Le caller orchestre lui-même la combinaison TeaCache × AttentionCache
  × CFGSkip × WA-RS — arsenal ne propose pas d'orchestrateur unifié.
- La couche kernel reste à faire (Metal block-sparse) ; les masques de
  l'étape 1 et la permutation de l'étape 6 sont les prérequis directs.

## Alternatives considérées

- **Bundler un orchestrateur unique** (un `SparseAttentionRunner` qui
  consomme tous les caches). Rejeté : trop opinionated sur la
  signature de la fonction d'attention, multiplie les chemins de test,
  et chaque port a sa propre boucle de denoising.
- **Implémenter le kernel Metal block-sparse en parallèle**. Rejeté :
  c'est un projet à part entière (Metal-FlashAttention équivalent),
  out-of-scope pour mid-level. Les primitives ici sont conçues pour
  s'intégrer à un kernel externe quand il sera disponible.
- **Schedule offline / calibration heuristics** pour choisir
  automatiquement les seuils. Rejeté : trop dépendant du modèle ;
  laissé au caller.

## Porte de sortie / révision

- Si un kernel Metal block-sparse devient disponible (philipturner/MFA
  ou autre), l'étape 6 (permutation) doit être validée contre la
  granularité de bloc effective de ce kernel — pas de changement
  d'API attendu mais nécessite un test d'intégration.
- Si plusieurs ports finissent par écrire la même boucle
  d'orchestration, envisager un `SparseAttentionRunner` extrait des
  ports les plus matures.

## Références

- [DiTFastAttn — Attention Sharing across Timesteps / CFG / Spatial](https://arxiv.org/abs/2406.08552)
- [Sparse VideoGen (ICML 2025)](https://arxiv.org/abs/2502.01776)
- [Sliding Tile Attention (ICML 2025)](https://arxiv.org/abs/2502.04507)
- [Radial Attention](https://arxiv.org/abs/2506.19852)
- [SVG2 — Semantic Permutation](https://arxiv.org/abs/2505.18875)
- TeaCache (Liu et al.) — déjà implémenté dans `mlx_arsenal.diffusion.TeaCacheController`.
