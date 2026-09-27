# ADR-0000: Adopt intendant

- **Status**: accepted
- **Date**: 2026-04-30
- **Stacks affected**: * (cross-cutting)

## Context

This repo adopts [intendant](https://github.com/dgrauet/intendant) as its
governance framework — handbook, audit (tier 2), scaffolder (tier 3). The
`.intendant.toml` file at the repo root declares the stack, the
enforcement mode applied, and the justified exemptions.

## Decision

- Stack detected at adoption: `python`
- Initial mode: `advisory` (findings are reported but block nothing).
- All future ADRs in the repo are numbered from 0001.

## Consequences

- The auditor (intendant tier 2) can run on this repo and report
  deviations from the baseline.
- Exemptions must be listed in `.intendant.toml` with a reason.

## Alternatives considered

- Adopt nothing (keep conventions implicit). Rejected: convention debt
  piles up silently.
- Adopt another framework: no multi-stack equivalent was identified at
  adoption time.

## Exit / revision

- If intendant stops keeping up with the tooling, switch to a permanent
  `mode = advisory` and take the standards back by hand.
- If a `v2` baseline breaks too many rules: freeze at `version = "1"` and
  plan a targeted migration.
