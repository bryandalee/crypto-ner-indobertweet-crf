"""Verify the vectorised significance tests match the naive implementation,
then benchmark them at the real corpus size (1,461 sentences, 9 systems)."""
import time
import numpy as np
import utils as U

rng = np.random.RandomState(7)

def make_corpus(n, n_tokens=22, ent_rate=0.55):
    y_true, preds = [], []
    for _ in range(n):
        labs = ["O"] * n_tokens
        if rng.rand() < ent_rate:
            s = rng.randint(0, n_tokens - 2)
            labs[s] = "B-CRYPTO"
            if rng.rand() < 0.25:
                labs[s + 1] = "I-CRYPTO"
        y_true.append(labs)
    return y_true

def corrupt(y_true, err=0.3):
    out = []
    for labs in y_true:
        t = list(labs)
        if rng.rand() < err:
            j = rng.randint(0, len(t))
            t[j] = rng.choice(["O", "B-CRYPTO", "I-CRYPTO"])
        out.append(t)
    return out

# --- 1. equivalence against a literal reference implementation --------------
print("1. Correctness vs naive reference")
y_true = make_corpus(120)
pa, pb = corrupt(y_true, 0.45), corrupt(y_true, 0.15)

def naive_paired_bootstrap(y_true, pred_a, pred_b, n_resamples, seed):
    r = np.random.RandomState(seed)
    n = len(y_true)
    obs = U.entity_f1(y_true, pred_b)["f1"] - U.entity_f1(y_true, pred_a)["f1"]
    d = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = r.randint(0, n, n)
        yt = [y_true[i] for i in idx]
        fa = U.prf(*U.entity_counts(yt, [pred_a[i] for i in idx]))[2]
        fb = U.prf(*U.entity_counts(yt, [pred_b[i] for i in idx]))[2]
        d[k] = fb - fa
    c = d - d.mean()
    return obs, float((np.abs(c) >= abs(obs)).mean())

# same seed + same block size as n_resamples => identical RNG draw order
obs_ref, p_ref = naive_paired_bootstrap(y_true, pa, pb, 300, seed=11)
res = U.paired_bootstrap(y_true, pa, pb, n_resamples=300, seed=11, block=300)
assert abs(res["delta_f1"] - obs_ref) < 1e-12, (res["delta_f1"], obs_ref)
assert abs(res["p_value"] - p_ref) < 1e-12, (res["p_value"], p_ref)
print(f"   paired_bootstrap identical to reference  (dF1={obs_ref:+.4f}, p={p_ref:.4f})")

# per-sentence counts must aggregate to the corpus-level metric
tp, npr, nt = U.per_sentence_counts(y_true, pa)
m = U.entity_f1(y_true, pa)
assert abs(float(U._f1_from_sums(tp.sum(), npr.sum(), nt.sum())) - m["f1"]) < 1e-12
assert (tp.sum(), npr.sum(), nt.sum()) == U.entity_counts(y_true, pa)
print(f"   per_sentence_counts aggregates to entity_f1 (F1={m['f1']:.4f})")

# blocking must not change the answer
r1 = U.paired_bootstrap(y_true, pa, pb, n_resamples=1000, seed=3, block=1000)
r2 = U.paired_bootstrap(y_true, pa, pb, n_resamples=1000, seed=3, block=250)
assert abs(r1["delta_f1"] - r2["delta_f1"]) < 1e-12
assert abs(r1["p_value"] - r2["p_value"]) < 0.02, (r1, r2)
print(f"   block size does not shift the result ({r1['p_value']:.4f} vs {r2['p_value']:.4f})")

# --- 2. sanity behaviour ----------------------------------------------------
print("\n2. Behaviour")
same = U.paired_bootstrap(y_true, pa, pa, n_resamples=1000, seed=1)
assert abs(same["delta_f1"]) < 1e-12 and same["p_value"] > 0.9
print(f"   identical systems -> dF1=0, p={same['p_value']:.3f}")

diff = U.paired_bootstrap(y_true, pa, pb, n_resamples=2000, seed=1)
ar = U.approximate_randomization(y_true, pa, pb, n_trials=2000, seed=1)
assert diff["p_value"] < 0.05 and ar["p_value"] < 0.05
print(f"   clearly different systems -> bootstrap p={diff['p_value']:.4f}, "
      f"randomization p={ar['p_value']:.4f}")

lo, hi = U.bootstrap_ci(y_true, pa, n_resamples=2000)
assert lo < U.entity_f1(y_true, pa)["f1"] < hi
print(f"   CI brackets the point estimate: [{lo:.3f}, {hi:.3f}]")

# --- 3. benchmark at real scale --------------------------------------------
print("\n3. Timing at the real corpus size (1,461 sentences)")
y_big = make_corpus(1461)
systems = {f"sys{i}": corrupt(y_big, 0.1 + 0.05 * i) for i in range(9)}
ref = systems["sys0"]

t0 = time.time()
_ = U.per_sentence_counts(y_big, ref)
t_counts = time.time() - t0
print(f"   per_sentence_counts (once per system) : {t_counts:.3f}s")

t0 = time.time()
U.bootstrap_ci(y_big, ref, n_resamples=2000)
t_ci = time.time() - t0

t0 = time.time()
U.paired_bootstrap(y_big, systems["sys3"], ref, n_resamples=2000)
t_pb = time.time() - t0
print(f"   bootstrap_ci     (2000 resamples)     : {t_ci:.2f}s")
print(f"   paired_bootstrap (2000 resamples)     : {t_pb:.2f}s")

t0 = time.time()
for name, pred in systems.items():
    U.bootstrap_ci(y_big, pred, n_resamples=2000)
    if name != "sys0":
        U.paired_bootstrap(y_big, pred, ref, n_resamples=2000)
total = time.time() - t0
print(f"\n   FULL CELL A EQUIVALENT (9 systems)    : {total:.1f}s")

t0 = time.time()
U.approximate_randomization(y_big, systems["sys3"], ref, n_trials=2000)
print(f"   approximate_randomization (2000)      : {time.time()-t0:.2f}s")

assert total < 120, f"still too slow: {total:.0f}s"
print("\nAll checks passed.")
