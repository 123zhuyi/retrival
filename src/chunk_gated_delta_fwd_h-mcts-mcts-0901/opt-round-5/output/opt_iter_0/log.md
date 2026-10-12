# Opt-Iter 0 — multibuffer=False probe

## Attempt
- Single-variable launch probe: changed `multibuffer=True` -> `multibuffer=False` on the verified champion anchor (all else byte-identical, nw=4, unit_flag=True).

## Why this variable
- R4's one untested-variable probe covered `num_warps 4->8` (regressed). The launch dict also carries `multibuffer` and `unit_flag`; `multibuffer` had never been A/B'd in the lineage (R2/R3/R4 all assumed True).

## Result
- verify: 8/8 passed (numerically fine).
- benchmark geomean: **6.8053x** vs anchor **7.2335x** → regression **-5.9%**.
- One frequency-drift sample (device 7 1800->800 MHz at sample #47 of 49) was observed on the last case; direction is consistent across the whole suite, so the regression is real, not noise.

## Decision
- **REVERT** to anchor (multibuffer=True). Record as new negative data point in operator_experience.md.
- improvement_made=false; target_reached=true (7.2335x >= 5x). Phase 4 exits with the anchor as final code.
