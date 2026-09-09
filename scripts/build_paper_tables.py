#!/usr/bin/env python3
"""Generate paper/tables/*.tex and paper/tables/macros.tex from results/.

Every number that appears in the manuscript is produced here from the CSV / JSON files that the
notebook writes.  Nothing is typed by hand.  Run from the repository root:

    python scripts/build_paper_tables.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

AGENT_LABELS = {
    "DES-UCB": "DES-UCB",
    "A1": "A1 no switch",
    "A2": "A2 no suppression",
    "A3": "A3 no detector",
    "A4": "A4 no BOB",
    "A5": "A5 decay instead",
    "A6": "A6 uncertainty gain",
    "A7": "A7 UCB1 switch",
    "B1": "B1 LinUCB",
    "B2": "B2 LinUCB warm",
    "B3": "B3 SW-UCB",
    "B4": "B4 BOB",
    "B5": "B5 D-LinUCB",
    "B6": "B6 PH-restart",
    "B7": "B7 Prior-only",
    "B8": "B8 Random",
    "B9": "B9 Oracle",
}
DRIFT_ORDER = ["none", "abrupt", "gradual", "recurring", "incremental", "sinusoidal"]
AGENT_ORDER = ["DES-UCB"] + [f"B{i}" for i in range(1, 10)]

macros: dict[str, str] = {}


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.{nd}f}"


_DIGITS = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}


def ag(a: str) -> str:
    """Agent id -> macro-safe token (DES-UCB -> DES, B2 -> BTwo)."""
    return re.sub(r"[0-9]", lambda m: _DIGITS[m.group()], a.replace("-UCB", "").replace("-", ""))


def macro(name: str, value) -> None:
    """Register a LaTeX macro.  Names must be letters only (LaTeX restriction)."""
    name = re.sub(r"[0-9]", lambda m: _DIGITS[m.group()], name)
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = str(value)


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_table(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body)


def bold_min(values: pd.Series, exclude=()) -> dict:
    """Return {agent: True} for the minimum among non-excluded agents."""
    v = values.drop([e for e in exclude if e in values.index])
    return {v.idxmin(): True} if len(v) else {}


# ---------------------------------------------------------------- grid (Table: synthetic grid)
grid = pd.read_csv(RES / "grid_summary.csv")
piv = grid.pivot(index="agent", columns="drift", values="dyn_regret_mean").reindex(AGENT_ORDER)
piv_sd = grid.pivot(index="agent", columns="drift", values="dyn_regret_std").reindex(AGENT_ORDER)
drifts = [d for d in DRIFT_ORDER if d in piv.columns]
lines = [
    r"\begin{tabular}{l" + "r" * len(drifts) + "}",
    r"\toprule",
    "Agent & " + " & ".join(drifts) + r" \\",
    r"\midrule",
]
for a in AGENT_ORDER:
    cells = []
    for d in drifts:
        best = bold_min(piv[d], exclude=("B9",))
        s = f"{fmt(piv.loc[a, d], 1)} $\\pm$ {fmt(piv_sd.loc[a, d], 1)}"
        cells.append(r"\textbf{" + s + "}" if best.get(a) else s)
    lines.append(tex_escape(AGENT_LABELS[a]) + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("grid", "\n".join(lines))
for d in drifts:
    for a in ("DES-UCB", "B2", "B4", "B6", "B3", "B5"):
        macro("grid" + d.capitalize() + ag(a), fmt(piv.loc[a, d], 1))
rec = grid.pivot(index="agent", columns="drift", values="recovery_time")
for d in ("abrupt", "recurring"):
    for a in ("DES-UCB", "B4", "B6", "B2"):
        macro("recovery" + d.capitalize() + ag(a), fmt(rec.loc[a, d], 0))

# ---------------------------------------------------------------- ablations
abl = pd.read_csv(RES / "ablation_summary.csv")
abl_drifts = [d for d in DRIFT_ORDER if d in set(abl.drift)]
abl_agents = ["DES-UCB"] + [f"A{i}" for i in range(1, 8)]
lines = [
    r"\begin{tabular}{l" + "rr" * len(abl_drifts) + "}",
    r"\toprule",
    "Variant & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for d in abl_drifts) + r" \\",
    " & " + " & ".join(["regret & $\\Delta$ [95\\% CI]"] * len(abl_drifts)) + r" \\",
    r"\midrule",
]
for a in abl_agents:
    cells = []
    for d in abl_drifts:
        r = abl[(abl.drift == d) & (abl.agent == a)]
        if r.empty:
            cells += ["--", "--"]
            continue
        r = r.iloc[0]
        cells.append(fmt(r.dyn_regret_mean, 1))
        if a == "DES-UCB":
            cells.append("--")
        else:
            cells.append(f"{fmt(r.DES_minus_this, 1)} [{fmt(r.ci_lo, 1)}, {fmt(r.ci_hi, 1)}]")
        macro("abl" + d.capitalize() + ag(a), fmt(r.dyn_regret_mean, 1))
        if a != "DES-UCB":
            macro("ablDiff" + d.capitalize() + ag(a), fmt(r.DES_minus_this, 1))
            macro("ablLo" + d.capitalize() + ag(a), fmt(r.ci_lo, 1))
            macro("ablHi" + d.capitalize() + ag(a), fmt(r.ci_hi, 1))
    lines.append(tex_escape(AGENT_LABELS[a]) + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("ablation", "\n".join(lines))

# ---------------------------------------------------------------- staleness sweep
sw = pd.read_csv(RES / "staleness_sweep.csv")
lines = [
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"Knob & value & DES-UCB & A2 & B2 & B4 & DES$-$A2 [95\% CI] \\",
    r"\midrule",
]
for _, r in sw.iterrows():
    val = f"{r.value:g}" + (" (pop.)" if r.tag == "pre_drift=F" and r.knob != "p_mismatch" else "")
    lines.append(
        f"{tex_escape(str(r.knob))} & {val} & {fmt(r['regret_DES-UCB'], 1)} & {fmt(r.regret_A2, 1)} & "
        f"{fmt(r.regret_B2, 1)} & {fmt(r.regret_B4, 1)} & {fmt(r.DES_minus_A2, 1)} "
        f"[{fmt(r.ci_lo, 1)}, {fmt(r.ci_hi, 1)}] \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("sweep", "\n".join(lines))
sig = sw[sw.ci_hi < 0]
macro("sweepRows", len(sw))
macro("sweepSigRows", len(sig))
m_pre = sw[(sw.knob == "magnitude") & (sw.tag == "pre_drift=T")].set_index("value")
for v in m_pre.index:
    macro("sweepMagPre" + f"{v:g}".replace(".", "p"), fmt(m_pre.loc[v, "DES_minus_A2"], 1))
sp = sw[sw.knob == "sigma_prior"].set_index("value")
macro("sweepSigmaPriorLarge", fmt(sp["DES_minus_A2"].iloc[-1], 1))
macro("sweepSigmaPriorLargeHi", fmt(sp["ci_hi"].iloc[-1], 1))
bb = sw[sw.knob == "burn_in_B"].set_index("value")
macro("sweepBurnZero", fmt(bb.loc[0.0, "DES_minus_A2"], 1))
macro("sweepBurnHundred", fmt(bb.loc[100.0, "DES_minus_A2"], 1))

# ---------------------------------------------------------------- phase diagram
ph = pd.read_csv(RES / "phase_diagram.csv", index_col=0)
lines = [
    r"\begin{tabular}{l" + "r" * len(ph.columns) + "}",
    r"\toprule",
    r"$\|\theta_1-\theta_0\|$ & " + " & ".join(f"$n_P={c}$" for c in ph.columns) + r" \\",
    r"\midrule",
]
for mag, row in ph.iterrows():
    lines.append(f"{mag:g} & " + " & ".join(fmt(v, 1) for v in row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("phase", "\n".join(lines))
macro("phaseMinDiff", fmt(ph.values.min(), 1))
macro("phaseMaxDiff", fmt(ph.values.max(), 1))
macro("phaseNegCells", int((ph.values < 0).sum()))
macro("phaseCells", int(ph.size))

# ---------------------------------------------------------------- rate
rate = pd.read_csv(RES / "rate_slopes.csv").set_index("agent")
cks = [c for c in rate.columns if c.startswith("R@")]
lines = [
    r"\begin{tabular}{lr" + "r" * len(cks) + "}",
    r"\toprule",
    "Agent & slope & " + " & ".join(f"$R({c[2:]})$" for c in cks) + r" \\",
    r"\midrule",
]
for a in ["DES-UCB", "B4", "B3"]:
    if a in rate.index:
        lines.append(
            tex_escape(AGENT_LABELS[a])
            + f" & {fmt(rate.loc[a, 'slope'], 3)} & "
            + " & ".join(fmt(rate.loc[a, c], 1) for c in cks)
            + r" \\"
        )
        macro("rateSlope" + ag(a), fmt(rate.loc[a, "slope"], 3))
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("rate", "\n".join(lines))
macro("rateCheckpoints", ", ".join(c[2:] for c in cks))

# ---------------------------------------------------------------- real data
real = pd.read_csv(RES / "real_summary.csv")
real_agents = ["DES-UCB", "A2", "B1", "B2", "B4", "B7", "B8"]
cols = ["cum_precision@5", "cum_precision@10", "cum_precision@20", "cum_precision@40", "offline_regret", "cum_recall@T"]
heads = ["P@5", "P@10", "P@20", "P@40", "off. regret", "recall@T"]
lines = [
    r"\begin{tabular}{llr" + "r" * len(cols) + "}",
    r"\toprule",
    "Data & prot. & agent & " + " & ".join(heads) + r" \\",
    r"\midrule",
]
for (ds, pr), g in real.groupby(["dataset", "protocol"], sort=False):
    g = g.set_index("agent").reindex([a for a in real_agents if a in set(g.agent)])
    for i, (a, r) in enumerate(g.iterrows()):
        lab = f"{ds} & {pr}" if i == 0 else " & "
        lines.append(f"{lab} & {tex_escape(AGENT_LABELS[a])} & " + " & ".join(fmt(r[c], 3) for c in cols) + r" \\")
        key = ds.replace("-", "").replace("100k", "hundredk").replace("1m", "onem") + pr + ag(a)
        macro("real" + key + "PFourty", fmt(r["cum_precision@40"], 3))
        macro("real" + key + "PTwenty", fmt(r["cum_precision@20"], 3))
        macro("real" + key + "Regret", fmt(r["offline_regret"], 2))
        macro("real" + key + "Recall", fmt(r["cum_recall@T"], 3))
        macro("real" + key + "PTen", fmt(r["cum_precision@10"], 3))
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
write_table("real", "\n".join(lines))
n_units = real.groupby("dataset")["n_units"].first()
macro("realUsersMLonem", int(n_units.get("ml-1m", 0)))
macro("realUsersMLhundredk", int(n_units.get("ml-100k", 0)))
macro("realUsersMind", int(n_units.get("mind", 0)))

# ---------------------------------------------------------------- pilot
pil = pd.read_csv(RES / "pilot_regret.csv").set_index("agent")
det = pd.read_csv(RES / "pilot_detection.csv").set_index("agent")
pv = json.loads((RES / "pilot_verdict.json").read_text())
lines = [
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"Agent & regret & detected & missed & mean delay & firings \\",
    r"\midrule",
]
for a in ["DES-UCB", "A2", "B4"]:
    d = det.loc[a]
    lines.append(
        f"{tex_escape(AGENT_LABELS[a])} & {fmt(pil.loc[a, 'dynamic_regret_mean'], 1)} $\\pm$ "
        f"{fmt(pil.loc[a, 'dynamic_regret_std'], 1)} & {int(d.detected)}/{int(d.n_changes)} & {int(d.missed)} & "
        f"{fmt(d.mean_delay, 1)} & {int(d.total_firings)} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("pilot", "\n".join(lines))
macro("pilotDelay", fmt(pv["detector_mean_delay"], 1))
macro("pilotDetected", fmt(100 * pv["detected_fraction"], 0))
macro("pilotBurnIn", int(pv["burn_in_B"]))
macro("pilotUnits", int(pil.loc["DES-UCB", "n_units"]))
macro("pilotFalseAlarms", fmt(det.loc["DES-UCB", "false_alarms_per_unit"], 2))
macro("pilotDESminusAtwo", fmt(pv["DES_minus_A2_regret"], 1))
macro("pilotDESminusBfour", fmt(pv["DES_minus_B4_regret"], 1))

# ---------------------------------------------------------------- claims
claims = json.loads((RES / "claims.json").read_text())
lines = [
    r"\begin{tabular}{llp{0.42\linewidth}l}",
    r"\toprule",
    r"Claim & Verdict & Test & Key statistic \\",
    r"\midrule",
]
KEY = {
    "sanity_oracle": lambda s: f"min regret {fmt(s['min_per_round_regret'], 2)}",
    "suppression": lambda s: f"$\\Delta$ {fmt(s['mean_diff'], 1)} [{fmt(s['ci_lo'], 1)}, {fmt(s['ci_hi'], 1)}]",
    "stale_ratio": lambda s: f"$\\Delta$ {fmt(s['mean_diff'], 3)} [{fmt(s['ci_lo'], 3)}, {fmt(s['ci_hi'], 3)}]",
    "not_redundant": lambda s: f"$\\Delta$ {fmt(s['mean_diff'], 1)} [{fmt(s['ci_lo'], 1)}, {fmt(s['ci_hi'], 1)}]",
    "suppression_mis": lambda s: f"$\\Delta$ {fmt(s['mean_diff'], 1)} [{fmt(s['ci_lo'], 1)}, {fmt(s['ci_hi'], 1)}]",
    "no_harm": lambda s: f"ratio {fmt(s['ratio'], 2)}",
    "rate": lambda s: f"slopes {fmt(s['slope_DES'], 3)} vs {fmt(s['slope_B4'], 3)}",
    "real_precision": lambda s: f"$\\Delta$ {fmt(s['mean_diff'], 3)} [{fmt(s['ci_lo'], 3)}, {fmt(s['ci_hi'], 3)}]",
}
n_sup = n_not = 0
for name, c in claims.items():
    v = c["verdict"]
    n_sup += v == "SUPPORTED"
    n_not += v == "NOT SUPPORTED"
    key = KEY[name](c["stats"]) if name in KEY and "reason" not in c["stats"] else "not run"
    desc = tex_escape(c["description"]).replace("`", "")
    desc = desc.replace(">=", "$\\ge$").replace("<=", "$\\le$").replace(" < ", " $<$ ").replace("@", "@\\,")
    lines.append(f"\\texttt{{{tex_escape(name)}}} & {v.lower()} & {desc} & {key} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("claims", "\n".join(lines))
macro("claimsSupported", n_sup)
macro("claimsNotSupported", n_not)
macro("claimsTotal", len(claims))
s = claims["suppression"]["stats"]
macro("supRegretDES", fmt(s["regret_DES"], 1))
macro("supRegretAtwo", fmt(s["regret_A2"], 1))
macro("supDiff", fmt(s["mean_diff"], 1))
macro("supLo", fmt(s["ci_lo"], 1))
macro("supHi", fmt(s["ci_hi"], 1))
macro("supUnits", int(s["n_units"]))
macro("supDelay", fmt(s["detector_mean_delay"], 1))
macro("supDetected", fmt(100 * s["detected_fraction"], 1))
s = claims["stale_ratio"]["stats"]
macro("staleDES", fmt(s["stale_DES"], 3))
macro("staleAtwo", fmt(s["stale_A2"], 3))
macro("staleDiff", fmt(s["mean_diff"], 3))
macro("staleLo", fmt(s["ci_lo"], 3))
macro("staleHi", fmt(s["ci_hi"], 3))
s = claims["not_redundant"]["stats"]
macro("redRegretDES", fmt(s["regret_DES"], 1))
macro("redRegretAfive", fmt(s["regret_A5_decay"], 1))
macro("redDiff", fmt(s["mean_diff"], 1))
macro("redLo", fmt(s["ci_lo"], 1))
macro("redHi", fmt(s["ci_hi"], 1))
s = claims["suppression_mis"]["stats"]
macro("misRegretDES", fmt(s["regret_DES"], 1))
macro("misRegretAtwo", fmt(s["regret_A2"], 1))
macro("misDiff", fmt(s["mean_diff"], 1))
macro("misLo", fmt(s["ci_lo"], 1))
macro("misHi", fmt(s["ci_hi"], 1))
s = claims["no_harm"]["stats"]
macro("harmRegretDES", fmt(s["regret_DES"], 2))
macro("harmRegretBtwo", fmt(s["regret_B2"], 2))
macro("harmRatio", fmt(s["ratio"], 2))
macro("harmFfraction", fmt(100 * s["F_fraction_DES"], 0))
s = claims["rate"]["stats"]
macro("rateDES", fmt(s["slope_DES"], 3))
macro("rateBfour", fmt(s["slope_B4"], 3))
s = claims["real_precision"]["stats"]
if "reason" not in s:
    macro("realPrecDES", fmt(s["cum_precision@40_DES"], 3))
    macro("realPrecBtwo", fmt(s["cum_precision@40_B2"], 3))
    macro("realPrecDiff", fmt(s["mean_diff"], 3))
    macro("realPrecLo", fmt(s["ci_lo"], 3))
    macro("realPrecHi", fmt(s["ci_hi"], 3))
    macro("realPrecUsers", int(s["n_users"]))

# ---------------------------------------------------------------- configuration constants (from configs/, not results)
import yaml  # noqa: E402

cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
for k in [
    "n_seeds",
    "n_users",
    "T",
    "d",
    "N_items",
    "K_cand",
    "lam",
    "beta",
    "S",
    "w_0",
    "w_min",
    "H",
    "bob_gamma",
    "burn_in_B",
]:
    macro("cfg" + re.sub(r"[^A-Za-z]", "", k.title()), f"{cfg[k]:g}" if isinstance(cfg[k], (int, float)) else cfg[k])
macro("cfgSwitchGamma", f"{cfg['switch']['gamma']:g}")
macro("cfgSwitchG", f"{cfg['switch']['G']:g}")
macro("cfgCandidateWindows", ", ".join(str(w) for w in cfg["candidate_windows"]))
macro("cfgPHThreshold", f"{cfg['detector']['lam_threshold']:g}")
macro("cfgPHDelta", f"{cfg['detector']['delta']:g}")
macro("cfgPHForget", f"{cfg['detector']['alpha_forget']:g}")
macro("cfgPHMin", f"{cfg['detector']['min_instances']:g}")
macro("cfgCohorts", cfg["prior"]["K_cohorts"])
macro("cfgPriorRows", cfg["prior"]["n_prior_rows"])
macro("cfgSigmaPrior", f"{cfg['prior']['sigma_prior']:g}")
macro("cfgSigmaReward", f"{cfg['drift']['sigma_reward']:g}")
macro("cfgChangePoints", ", ".join(str(c) for c in cfg["drift"]["change_points"]))
macro("cfgMagnitude", f"{cfg['drift']['magnitude']:g}")
macro("cfgGradualG", cfg["drift"]["gradual_G"])
macro("cfgRecurringP", cfg["drift"]["recurring_P"])
macro("cfgIncrementalEps", f"{cfg['drift']['incremental_eps']:g}")
macro("cfgSinePeriod", cfg["drift"]["sinusoidal_period"])
macro("cfgSWwindow", cfg["baselines"]["sw_window"])
macro("cfgDiscount", f"{cfg['baselines']['discount_gamma']:g}")
ex = cfg["experiments"]
for name in ["pilot", "grid", "ablation", "sweep", "phase", "rate", "claims"]:
    e = ex[name]
    macro("exp" + name.capitalize() + "Seeds", e["n_seeds"])
    macro("exp" + name.capitalize() + "Users", e["n_users"])
    macro("exp" + name.capitalize() + "T", e["T"])
macro("expPilotChangePoint", ex["pilot"]["change_points"][0])
ml = yaml.safe_load((ROOT / "configs" / "movielens_1m.yaml").read_text())
macro("mlD", ml["d"])
macro("mlSvd", ml["svd_dim"])
macro("mlMinRatings", ml["min_ratings"])
macro("mlT", ml["T"])
macro("mlK", ml["K_cand"])
macro("mlHit", ml["hit_threshold"])
macro("mlMaxUsers", ml["max_test_users"])
macro("mlPmis", f"{ml['protocol_c_p_mismatch']:g}")
macro("mlH", ml["H"])
macro("mlBurn", ml["burn_in_B"])
macro("mlWzero", ml["w_0"])
macro("mlWmin", ml["w_min"])
macro("mlWindows", ", ".join(str(w) for w in ml["candidate_windows"]))
mind = yaml.safe_load((ROOT / "configs" / "mind_small.yaml").read_text())
macro("mindSvd", mind["svd_dim"])
macro("mindT", mind["T"])
macro("mindMinImp", mind["min_impressions"])
macro("mindMaxUsers", mind["max_test_users"])
macro("mindH", mind["H"])
macro("mindBurn", mind["burn_in_B"])

# ---------------------------------------------------------------- write macros
lines = ["% Auto-generated by scripts/build_paper_tables.py from results/ and configs/.  Do not edit."]
for k, v in sorted(macros.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(OUT / "macros.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {len(macros)} macros and {len(list(OUT.glob('*.tex'))) - 1} tables to {OUT}")
