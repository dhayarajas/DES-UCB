# Results skeleton (auto-generated from results/claims.json)

Only the bullets under *Supported* may appear as findings in the paper.

## Supported
- **sanity_oracle** — oracle expected reward >= every agent's expected reward, every round. Numbers: min_per_round_regret=0, regret_B9=0, regret_min_other=50.44
- **suppression** — dyn_regret[DES] < dyn_regret[A2 no_suppression] on abrupt drift with a stale (pre-drift) prior. Numbers: regret_DES=50.44, regret_A2=65.12, mean_diff=-14.68, ci_lo=-15.95, ci_hi=-13.43, n_units=500, detector_mean_delay=23.32, detected_fraction=0.998
- **stale_ratio** — post-change rounds conditioned on P: DES < A2. Numbers: stale_DES=0.0704, stale_A2=0.09871, mean_diff=-0.02832, ci_lo=-0.03248, ci_hi=-0.02455
- **suppression_mis** — same as `suppression` under p_mismatch=0.2 with a population (not pre-drift) prior. Numbers: regret_DES=78.56, regret_A2=93.3, mean_diff=-14.73, ci_lo=-16.02, ci_hi=-13.45
- **rate** — log-log regret slope on sinusoidal drift: DES <= B4 + 0.05. Numbers: slope_DES=1.168, slope_B4=1.207, checkpoints=[250, 500, 1000, 2000, 4000]

## Not supported / do not write
- **not_redundant** [NOT SUPPORTED] — purge+reset+burn-in beats passive decay of the prior (A5). Numbers: regret_DES=50.44, regret_A5_decay=47.26, mean_diff=3.176, ci_lo=1.904, ci_hi=4.353
- **no_harm** [NOT SUPPORTED] — dyn_regret[DES] <= 1.10 * dyn_regret[B2 warm LinUCB] with no drift. Numbers: regret_DES=11.14, regret_B2=1.622, ratio=6.864, F_fraction_DES=0.5161
- **real_precision** [NOT SUPPORTED] — ML-1M protocol B: cum_precision@40[DES] >= cum_precision@40[B2]. Numbers: cum_precision@40_DES=0.7195, cum_precision@40_B2=0.7497, mean_diff=-0.03017, ci_lo=-0.03517, ci_hi=-0.02491, n_users=295

## Decision table
- NO DECISION-TABLE BRANCH TRIGGERED (claims consistent with the title as stated)