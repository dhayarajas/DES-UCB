import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import des_ucb_core as C  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return C.load_config()


# ---------------------------------------------------------------- estimators
def test_ridge_add_remove_matches_exact_recompute():
    rng = np.random.default_rng(0)
    d = 6
    est = C.RidgeUCB(d, lam=1.0, beta=0.5, S=100.0)
    X = rng.standard_normal((40, d)); r = rng.standard_normal(40)
    for x, y in zip(X, r):
        est.add(x, y)
    for x, y in zip(X[:15], r[:15]):
        est.remove(x, y)
    kept = slice(15, 40)
    V = np.eye(d) + X[kept].T @ X[kept]
    b = X[kept].T @ r[kept]
    np.testing.assert_allclose(est.theta(), np.linalg.solve(V, b), atol=1e-8)
    np.testing.assert_allclose(est.V_inv, np.linalg.inv(V), atol=1e-8)


def test_ridge_clips_theta_to_ball():
    est = C.RidgeUCB(3, lam=1e-3, beta=0.5, S=1.0)
    for _ in range(50):
        est.add(np.array([1.0, 0, 0]), 100.0)
    assert np.linalg.norm(est.theta()) <= 1.0 + 1e-9


def test_feedback_eviction_on_shrink_and_purge():
    d = 4
    rng = np.random.default_rng(1)
    F = C.FeedbackEstimator(d, window=100)
    for t in range(50):
        F.add_live(t, rng.standard_normal(d), 1.0)
    assert len(F) == 50
    F.set_window(10, t=49)               # keep s > 49-10 = 39 -> s in 40..49
    assert len(F) == 10 and F.rows[0][0] == 40
    # exact recompute against the kept rows
    X = np.stack([x for _, x, _ in F.rows]); r = np.array([y for _, _, y in F.rows])
    np.testing.assert_allclose(F.theta(), np.linalg.solve(np.eye(d) + X.T @ X, X.T @ r), atol=1e-8)
    # shrinking only changes the ACTIVE window: growing it again restores the retained rows exactly
    F.set_window(30, t=49)
    assert len(F) == 30 and F.rows[0][0] == 20
    X = np.stack([x for _, x, _ in F.rows]); r = np.array([y for _, _, y in F.rows])
    np.testing.assert_allclose(F.theta(), np.linalg.solve(np.eye(d) + X.T @ X, X.T @ r), atol=1e-8)
    F.set_window(10, t=49)
    assert len(F) == 10 and F.rows[0][0] == 40
    # an explicit drift purge is irreversible
    F.purge_older_than(45)
    assert len(F) == 5 and F.rows[0][0] == 45
    F.set_window(100, t=49)
    assert len(F) == 5 and F.rows[0][0] == 45
    F.set_window(10, t=49)
    # rolling eviction
    for t in range(50, 60):
        F.add_live(t, rng.standard_normal(d), 0.0)
    assert len(F) == 10 and F.rows[0][0] == 50
    assert len(F.history) == 15 and F.history[0][0] == 45  # retained (max_history=100), not active


def test_feedback_history_bounded_by_max_history():
    F = C.FeedbackEstimator(3, window=5, max_history=20)
    for t in range(100):
        F.add_live(t, np.ones(3), 1.0)
    assert len(F) == 5 and len(F.history) == 20
    F.set_window(50, t=99)
    assert len(F) == 20 and F.rows[0][0] == 80


# ---------------------------------------------------------------- switch
def test_switch_forced_feedback_exactly_B_rounds():
    sw = C.EvidenceSwitch("exp3", rng=np.random.default_rng(0))
    sw.force_feedback(7)
    picks = []
    for t in range(20):
        picks.append(sw.select(t))
        sw.update(picks[-1], 0.0)
        sw.tick()
    assert picks[:7] == ["F"] * 7
    # after the burn-in, EXP3 must be able to pick P again (gamma-exploration guarantees it)
    assert sw.burn_in_remaining == 0
    assert sw.probs()[0] > 0


def test_exp3_probabilities_sum_to_one_and_favour_better_source():
    rng = np.random.default_rng(0)
    sw = C.EvidenceSwitch("exp3", gamma=0.1, G=1.0, rng=rng)
    for t in range(500):
        m = sw.select(t)
        gain = 0.8 if m == "F" else -0.8   # F is the better source
        sw.update(m, gain + 0.1 * rng.standard_normal())
        sw.tick()
        assert abs(sw.probs().sum() - 1.0) < 1e-9
    p = sw.probs()
    assert p[1] > 0.8 and p[1] > p[0]


def test_ucb1_switch_favours_better_source():
    sw = C.EvidenceSwitch("ucb1", c=1.0, rng=np.random.default_rng(0))
    picks = []
    for t in range(500):
        m = sw.select(t)
        sw.update(m, 0.8 if m == "P" else -0.8)
        sw.tick()
        picks.append(m)
    assert picks[-100:].count("P") > 90


def test_reset_source_removes_credit():
    sw = C.EvidenceSwitch("exp3", gamma=0.1, rng=np.random.default_rng(0))
    for t in range(300):
        m = sw.select(t); sw.update(m, 1.0 if m == "P" else -1.0); sw.tick()
    assert sw.probs()[0] > 0.5
    sw.reset_source("P")
    assert sw.probs()[0] <= 0.5 + 1e-9


# ---------------------------------------------------------------- detectors
def test_page_hinkley_fires_on_mean_shift_not_on_noise():
    rng = np.random.default_rng(0)
    ph = C.PageHinkley(delta=0.005, lam_threshold=5.0, alpha_forget=0.999, min_instances=30)
    fired_stationary = [ph.update(abs(0.1 * rng.standard_normal())) for _ in range(2000)]
    assert not any(fired_stationary)
    ph = C.PageHinkley(delta=0.005, lam_threshold=5.0, alpha_forget=0.999, min_instances=30)
    for _ in range(500):
        ph.update(abs(0.1 * rng.standard_normal()))
    fired = [ph.update(1.0 + abs(0.1 * rng.standard_normal())) for _ in range(200)]
    assert any(fired)
    assert fired.index(True) < 50


def test_adwin_fires_on_shift():
    rng = np.random.default_rng(0)
    ad = C.ADWIN(delta=0.002)
    assert not any(ad.update(0.01 * rng.standard_normal()) for _ in range(300))
    assert any(ad.update(1.0 + 0.01 * rng.standard_normal()) for _ in range(100))


# ---------------------------------------------------------------- BOB
def test_bob_favours_better_window():
    bob = C.BOBWindowSelector([50, 100, 200], H=10, gamma=0.1, rng=np.random.default_rng(0))
    for _ in range(300):
        w = bob.pick()
        bob.update(w, 1.0 if w == 100 else 0.2)
    assert bob.probs()[1] > 0.7
    assert abs(bob.probs().sum() - 1) < 1e-9


class _NeverFires:
    def update(self, e):
        return False


class _SpyBOB:
    """Records (window, block_reward) pairs; always picks the next window in a fixed cycle."""

    def __init__(self, windows):
        self.windows, self.i, self.updates = list(windows), 0, []

    def pick(self):
        w = self.windows[self.i % len(self.windows)]
        self.i += 1
        return w

    def update(self, w, g):
        self.updates.append((w, g))


@pytest.mark.parametrize("agent_name", ["DES-UCB", "B4"])
def test_bob_block_credits_only_rewards_under_its_window(cfg, agent_name):
    cfg = C._deep_update(cfg, {"H": 10, "candidate_windows": [5, 20], "w_0": 8, "burn_in_B": 0})
    ag = C.make_agent(agent_name, cfg["d"], (np.zeros((1, cfg["d"])), np.zeros(1)), cfg, seed=0)
    spy = _SpyBOB(cfg["candidate_windows"])
    ag.bob = spy
    H = cfg["H"]
    est = ag.F if agent_name == "DES-UCB" else ag.est
    if agent_name == "DES-UCB":
        ag.detector = _NeverFires()  # the per-block reward jumps below would otherwise trigger drift
    rng = np.random.default_rng(0)
    windows_seen = []
    for t in range(4 * H):
        X = rng.standard_normal((3, cfg["d"]))
        idx = ag.step(t, X)
        windows_seen.append(est.window)
        ag.observe(t, X[idx], float(t // H))  # constant reward per block => unambiguous attribution
    # block 0 runs under w_0; each later block runs entirely under the window picked at its start
    assert windows_seen[:H] == [cfg["w_0"]] * H
    for k in range(1, 4):
        assert len(set(windows_seen[k * H:(k + 1) * H])) == 1
    assert len(spy.updates) == 3
    lo, hi = cfg.get("reward_range", (-2.0, 2.0))
    for k, (w, g) in enumerate(spy.updates, start=1):
        assert w == windows_seen[k * H]
        assert g == pytest.approx(float(np.clip((k - lo) / (hi - lo), 0, 1)))


def test_drift_mid_block_discards_bob_block(cfg):
    cfg = C._deep_update(cfg, {"H": 10, "candidate_windows": [5, 20], "w_0": 8, "w_min": 2, "burn_in_B": 0})
    ag = C.make_agent("DES-UCB", cfg["d"], (np.zeros((1, cfg["d"])), np.zeros(1)), cfg, seed=0)
    ag.bob = spy = _SpyBOB(cfg["candidate_windows"])
    ag.detector = _NeverFires()
    H = cfg["H"]
    rng = np.random.default_rng(0)
    windows_seen = []
    for t in range(3 * H):
        X = rng.standard_normal((3, cfg["d"]))
        idx = ag.step(t, X)
        windows_seen.append(ag.F.window)
        ag.observe(t, X[idx], 1.0)
        if t == H + 4:
            ag.on_drift(t)  # window changes mid-block -> that block must not be credited to anyone
    assert windows_seen[H + 4] != windows_seen[H + 5]
    assert len(spy.updates) == 1  # block 2 only (block 1 discarded); block 0 ran under w_0
    assert spy.updates[0][0] == windows_seen[2 * H] and len(set(windows_seen[2 * H:3 * H])) == 1


# ---------------------------------------------------------------- agent vs B2 on stationary env
@pytest.mark.xfail(
    strict=True,
    reason="Preregistered no-harm check. Measured at default config (sigma_reward=0.1, EXP3 gamma=0.1): "
    "DES-UCB ~2.3x B2 dynamic regret on stationary data, because the switch keeps sampling the "
    "sliding-window F source (>=gamma/2 exploration) while B2 pools prior+all live rows. "
    "Reported as NOT SUPPORTED in claims.json rather than tuned away.",
)
def test_des_within_10pct_of_warm_linucb_on_no_drift(cfg):
    cfg = C.load_config("drift_none", overrides={"T": 400})
    df = C.run(lambda s, u: C.SyntheticEnv(cfg, s, u), ["DES-UCB", "B2"], cfg,
               n_seeds=1, n_users=8, n_jobs=1, progress=False)
    dr = C.dynamic_regret(df)["dynamic_regret_mean"]
    assert dr["DES-UCB"] <= 1.10 * dr["B2"] + 1e-9, dr.to_dict()


def test_synthetic_oracle_dominates_every_round(cfg):
    cfg = C.load_config(overrides={"T": 300})
    df = C.run(lambda s, u: C.SyntheticEnv(cfg, s, u), ["DES-UCB", "B1", "B9"], cfg,
               n_seeds=1, n_users=3, n_jobs=1, progress=False)
    # oracle is on expected reward; check with the noise-free candidate means
    env = C.SyntheticEnv(cfg, 0, 0)
    for t in range(cfg["T"]):
        mu = env._means(t)
        assert env.best(t) >= mu.max() - 1e-12
    assert (df["regret_kind"].astype(str) == "dynamic").all()


def test_agent_flags_and_factory(cfg):
    env = C.SyntheticEnv(cfg, 0, 0)
    for name in ["DES-UCB"] + C.ALL_ABLATIONS + C.ALL_BASELINES:
        a = C.make_agent(name, env.d, env.D_prior, cfg, seed=0)
        if isinstance(a, C.OracleAgent):
            a.env = env
        for t in range(5):
            X = env.candidates(t); i = a.step(t, X); a.observe(t, X[i], env.reward(t, i))
    a = C.make_agent("A7", env.d, env.D_prior, cfg)
    assert a.switch.kind == "ucb1"


def test_drift_suppression_purges_and_forces_F(cfg):
    env = C.SyntheticEnv(cfg, 0, 0)
    a = C.DESUCBAgent(env.d, env.D_prior, cfg, C.Flags(no_detector=True, no_bob=True))
    for t in range(300):
        X = env.candidates(t); i = a.step(t, X); a.observe(t, X[i], env.reward(t, i))
    n_before = len(a.F)
    a.on_drift(299)
    w = max(cfg["w_min"], cfg["w_0"] // 2)
    assert len(a.F) <= a.window and a.window == w
    assert len(a.F) < n_before
    # purge boundary matches the active window: nothing older than 299 - w + 1 survives, even in history
    assert a.F.rows[0][0] == 299 - w + 1 and a.F.history[0][0] == 299 - w + 1
    a.F.set_window(cfg["w_0"], 299)
    assert len(a.F) == w and a.F.rows[0][0] == 299 - w + 1
    a.F.set_window(w, 299)
    for t in range(200, 200 + cfg["burn_in_B"]):
        X = env.candidates(t); i = a.step(t, X)
        assert a.source == "F"
        a.observe(t, X[i], env.reward(t, i))


# ---------------------------------------------------------------- real-data env safety
def test_rated_env_never_exposes_unrated_item():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((100, 5))
    rated = np.array([1, 5, 9, 13, 20, 33, 47, 50])
    rew = np.array([1, 0, 1, 0, 1, 1, 0, 0], float)
    env = C.RatedItemEnv(feats, rated, rew, T=4, K_cand=3, D_prior=None, seed=0)
    shown = []
    for t in range(env.T):
        X = env.candidates(t)
        assert X.shape[0] <= 3
        for row in X:  # every candidate is a rated item
            assert any(np.allclose(row, feats[i]) for i in rated)
        r = env.reward(t, 0)
        assert r in (0.0, 1.0)
        assert env.best(t) >= r
        shown.append(env.rated[env._pool_for(t)[0]])
    assert len(set(shown)) == len(shown)  # never re-shown
    env2 = C.RatedItemEnv(feats, rated, rew, T=4, K_cand=3, D_prior=None, seed=0)
    env2.rated_set = set()  # simulate corruption: any reveal must raise
    with pytest.raises(RuntimeError):
        env2.reward(0, 0)
    assert env.regret_kind == "offline"


def test_metric_namespaces_never_mix():
    dyn = pd.DataFrame({"seed": [0, 0], "user": [0, 0], "t": [0, 1], "agent": ["A", "A"], "reward": [1.0, 0.5],
                        "oracle_or_best_available": [1.0, 1.0], "regret_kind": ["dynamic", "dynamic"],
                        "regret": [0.0, 0.5], "source": ["P", "F"], "window": [1, 1],
                        "drift_fired": [False, False], "true_change": [False, False]})
    off = dyn.assign(regret_kind="offline")
    rep = dyn.assign(regret_kind="replay", counted=True)
    C.dynamic_regret(dyn); C.offline_regret(off); C.replay_reward(rep); C.cum_precision(off, ks=(1,))
    with pytest.raises(ValueError):
        C.dynamic_regret(off)
    with pytest.raises(ValueError):
        C.offline_regret(dyn)
    with pytest.raises(ValueError):
        C.cum_precision(dyn, ks=(1,))
    with pytest.raises(ValueError):
        C.replay_reward(dyn.assign(counted=True))
    with pytest.raises(ValueError):
        C.dynamic_regret(pd.concat([dyn, off]))


def test_cum_recall_requires_ordinal_user_index():
    df = pd.DataFrame({"seed": 0, "user": [0, 0, 1, 1], "agent": "A", "reward": [1.0, 1.0, 0.0, 1.0],
                       "regret_kind": "offline"})
    ds = C.RatedDataset("x", np.zeros((1, 1)), {}, [1001, 1002], {}, {}, {}, pd.DataFrame(), 1,
                        n_relevant={1001: 4, 1002: 2})
    rec = C.cum_recall(df, ds.n_relevant_by_ordinal())
    assert rec["A"] == pytest.approx((2 / 4 + 1 / 2) / 2)
    with pytest.raises(KeyError):
        C.cum_recall(df, pd.Series(ds.n_relevant))  # raw ids do not match the log's ordinals


def test_download_refuses_unverified_tls(tmp_path, monkeypatch):
    import requests

    def boom(*a, **k):
        assert k.get("verify", True) is True
        raise requests.exceptions.SSLError("bad cert")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(RuntimeError, match="verification failed"):
        C.download("https://example.invalid/x.zip", tmp_path / "x.zip")
    assert not (tmp_path / "x.zip").exists() and not (tmp_path / "x.zip.part").exists()


def test_replay_env_counts_only_matched_rounds():
    rng = np.random.default_rng(0)
    ctx = [rng.standard_normal((4, 3)) for _ in range(50)]
    a = rng.integers(4, size=50); r = rng.integers(2, size=50).astype(float)
    env = C.ReplayEnv(ctx, a, r)
    df = C._simulate_one(env, C.RandomAgent(3, None, {"seed": 0}), 0, 0, "B8")
    assert df["counted"].sum() < 50
    assert df.loc[df["counted"], "reward"].notna().all()
    assert (df["regret_kind"] == "replay").all()
    C.replay_reward(df)


def test_paired_bootstrap_ci():
    a = np.arange(100, dtype=float); b = a - 1.0
    m, lo, hi = C.paired_bootstrap_ci(a, b)
    assert abs(m - 1.0) < 1e-12 and lo <= 1.0 <= hi
    reg = C.ClaimRegistry(Path("/tmp/_claims_test.json"))
    assert reg.assert_claim("x", lo > 0, {"m": m}) is True
    assert reg.verdict("x") is True
