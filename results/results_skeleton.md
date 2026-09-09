# Results skeleton (auto-generated from results/claims.json)

Only the bullets under *Supported* may appear as findings in the paper.

## Supported
- **sanity_oracle** — oracle expected reward >= every agent's expected reward, every round. Numbers: min_per_round_regret=0, regret_B9=0, regret_min_other=50.38
- **suppression** — dyn_regret[DES] < dyn_regret[A2 no_suppression] on abrupt drift with a stale (pre-drift) prior. Numbers: regret_DES=50.38, regret_A2=65.12, mean_diff=-14.74, ci_lo=-16.05, ci_hi=-13.51, n_units=500, detector_mean_delay=22.68, detected_fraction=0.997
- **stale_ratio** — post-change rounds conditioned on P: DES < A2. Numbers: stale_DES=0.07043, stale_A2=0.09871, mean_diff=-0.02829, ci_lo=-0.03244, ci_hi=-0.02452
- **suppression_mis** — same as `suppression` under p_mismatch=0.2 with a population (not pre-drift) prior. Numbers: regret_DES=78.54, regret_A2=93.3, mean_diff=-14.75, ci_lo=-16.05, ci_hi=-13.46
- **rate** — log-log regret slope on sinusoidal drift: DES <= B4 + 0.05. Numbers: slope_DES=1.164, slope_B4=1.207, checkpoints=[250, 500, 1000, 2000, 4000]

## Not supported / do not write
- **not_redundant** [NOT SUPPORTED] — purge+reset+burn-in beats passive decay of the prior (A5). Numbers: regret_DES=50.38, regret_A5_decay=47.26, mean_diff=3.112, ci_lo=1.829, ci_hi=4.316
- **no_harm** [NOT SUPPORTED] — dyn_regret[DES] <= 1.10 * dyn_regret[B2 warm LinUCB] with no drift. Numbers: regret_DES=11.14, regret_B2=1.622, ratio=6.863, F_fraction_DES=0.5161
- **real_precision** [NOT SUPPORTED] — ML-1M protocol B: cum_precision@40[DES] >= cum_precision@40[B2]. Numbers: cum_precision@40_DES=0.7185, cum_precision@40_B2=0.7497, mean_diff=-0.03119, ci_lo=-0.03644, ci_hi=-0.02593, n_users=295

## Decision table
- NO DECISION-TABLE BRANCH TRIGGERED (claims consistent with the title as stated)