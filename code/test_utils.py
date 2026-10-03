"""Sanity tests for utils.py — run before trusting any experiment output."""
import numpy as np
import utils as U

ok = lambda name: print(f"  PASS  {name}")

# --- entity extraction ------------------------------------------------------
assert U.extract_entities(["O", "B-CRYPTO", "I-CRYPTO", "O"]) == [(1, 3, "CRYPTO")]
assert U.extract_entities(["B-CRYPTO", "B-CRYPTO"]) == [(0, 1, "CRYPTO"), (1, 2, "CRYPTO")]
assert U.extract_entities(["I-CRYPTO", "O"]) == [(0, 1, "CRYPTO")]      # stray I- tolerated
assert U.extract_entities(["O", "O"]) == []
assert U.extract_entities(["B-CRYPTO"]) == [(0, 1, "CRYPTO")]           # entity at sequence end
ok("extract_entities")

# --- entity-level metrics ---------------------------------------------------
yt = [["O", "B-CRYPTO", "O"], ["B-CRYPTO", "I-CRYPTO"], ["O", "O"]]
yp = [["O", "B-CRYPTO", "O"], ["B-CRYPTO", "O"], ["B-CRYPTO", "O"]]
m = U.entity_f1(yt, yp)
# gold: 2 entities; pred: 3; exact match: 1 -> P=1/3, R=1/2
assert abs(m["precision"] - 1/3) < 1e-9 and abs(m["recall"] - 0.5) < 1e-9
assert abs(m["f1"] - 0.4) < 1e-9
assert U.entity_f1(yt, yt)["f1"] == 1.0
ok("entity_f1 (exact match, partial span counts as wrong)")

# --- BIO violations ---------------------------------------------------------
bad, tot = U.count_bio_violations([["O", "I-CRYPTO"], ["B-CRYPTO", "I-CRYPTO"]])
assert bad == 1 and tot == 4
ok("count_bio_violations")

# --- folds are deterministic and cover every sample exactly once -------------
splits, fold_of = U.get_folds(1461)
assert len(splits) == 5
covered = np.concatenate([te for _, te in splits])
assert sorted(covered.tolist()) == list(range(1461))
assert U.get_folds(1461)[1].tolist() == fold_of.tolist()
train_sizes = [len(tr) for tr, _ in splits]
assert all(len(set(tr) & set(te)) == 0 for tr, te in splits)
ok(f"get_folds (reproducible, disjoint; train sizes {train_sizes})")

# --- gazetteer + denoising --------------------------------------------------
unamb, amb = {"bitcoin", "ethereum"}, {"ada", "sol"}
toks = "saya serok ADA kemarin".split()
den = U.gazetteer_label_tokens(toks, unamb, amb, denoise=True)
nai = U.gazetteer_label_tokens(toks, unamb, amb, denoise=False)
assert den[2] == "B-CRYPTO" and nai[2] == "B-CRYPTO"        # uppercase + financial ctx

toks2 = "tidak ada uang sama sekali".split()
den2 = U.gazetteer_label_tokens(toks2, unamb, amb, denoise=True)
nai2 = U.gazetteer_label_tokens(toks2, unamb, amb, denoise=False)
assert den2[1] == "O", "heuristic must suppress lowercase 'ada' with no financial context"
assert nai2[1] == "B-CRYPTO", "naive matching must label it (this is the -HD contrast)"
ok("gazetteer denoising suppresses ambiguous tokens; naive does not")

toks3 = "beli bitcoin sekarang".split()
assert U.gazetteer_label_tokens(toks3, unamb, amb)[1] == "B-CRYPTO"
assert U.gazetteer_label_tokens(["$BTC", "naik"], {"btc"}, set())[0] == "B-CRYPTO"
assert U.gazetteer_label_tokens(["$100", "naik"], {"btc"}, set())[0] == "O"   # price, not ticker
ok("gazetteer cashtag handling")

# --- significance tests -----------------------------------------------------
rng = np.random.RandomState(0)
n = 300
y_true = [["B-CRYPTO", "O"] if rng.rand() < .5 else ["O", "O"] for _ in range(n)]
y_good = [list(s) for s in y_true]
y_bad = []
for s in y_true:                       # corrupt ~30% of sentences
    t = list(s)
    if rng.rand() < .3:
        t[0] = "O" if t[0] != "O" else "B-CRYPTO"
    y_bad.append(t)

bs = U.paired_bootstrap(y_true, y_bad, y_good, n_resamples=400, seed=1)
assert bs["delta_f1"] > 0 and bs["p_value"] < 0.05, bs
ar = U.approximate_randomization(y_true, y_bad, y_good, n_trials=400, seed=1)
assert ar["p_value"] < 0.05, ar
same = U.paired_bootstrap(y_true, y_good, y_good, n_resamples=200, seed=1)
assert abs(same["delta_f1"]) < 1e-12 and same["p_value"] > 0.5
ok(f"paired_bootstrap (dF1={bs['delta_f1']:.3f}, p={bs['p_value']:.4f}) and randomization")

lo, hi = U.bootstrap_ci(y_true, y_bad, n_resamples=300)
assert lo < U.entity_f1(y_true, y_bad)["f1"] < hi
ok(f"bootstrap_ci [{lo:.3f}, {hi:.3f}]")

adj, rej = U.holm_bonferroni(np.array([0.001, 0.02, 0.04, 0.5]))
assert adj[0] < adj[1] <= adj[2] <= adj[3] and adj.max() <= 1.0
ok(f"holm_bonferroni {np.round(adj, 4).tolist()}")

# --- inter-annotator agreement ----------------------------------------------
a1 = [["O", "B-CRYPTO"], ["B-CRYPTO", "O"]]
a2 = [["O", "B-CRYPTO"], ["B-CRYPTO", "O"]]
a3 = [["O", "B-CRYPTO"], ["O", "O"]]
assert abs(U.fleiss_kappa(U.token_kappa_table([a1, a2, a1])) - 1.0) < 1e-9
k = U.fleiss_kappa(U.token_kappa_table([a1, a2, a3]))
assert 0 < k < 1
pw = U.pairwise_entity_agreement([a1, a2, a3])
assert pw["A1-A2"] == 1.0 and pw["A1-A3"] < 1.0 and len(pw) == 3
ok(f"fleiss_kappa={k:.4f}, pairwise entity F1 {({a: round(b,3) for a,b in pw.items()})}")

# --- corpus statistics ------------------------------------------------------
sents = [["beli", "Bitcoin"], ["harga", "ETH", "naik"], ["tidak", "ada", "koin"]]
labs = [["O", "B-CRYPTO"], ["O", "B-CRYPTO", "O"], ["O", "O", "O"]]
st = U.corpus_statistics(sents, labs)
assert st["sentences"] == 3 and st["entities"] == 2 and st["sentences_without_entity"] == 1
assert st["unique_surface_forms"] == 2 and st["hapax_ratio"] == 1.0
ok("corpus_statistics")

strata = U.assign_strata(sents, {"bitcoin", "eth", "pepe"}, set())
assert strata[0] == "Blue-Chip" and strata[1] == "Blue-Chip"
ok(f"assign_strata {strata}")

# --- model forward pass (CRF and softmax heads) -----------------------------
import torch
for use_crf in (True, False):
    torch.manual_seed(0)
    m = U.TokenClassifier.__new__(U.TokenClassifier)
    torch.nn.Module.__init__(m)
    m.use_crf, m.num_labels = use_crf, 3
    m.classifier = torch.nn.Linear(8, 3)
    m.dropout = torch.nn.Dropout(0.0)
    if use_crf:
        from torchcrf import CRF
        m.crf = CRF(num_tags=3, batch_first=True)
    else:
        m.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    emissions = m.classifier(torch.randn(2, 5, 8))
    mask = torch.ones(2, 5, dtype=torch.bool)
    labels = torch.tensor([[0, 1, 2, 0, 0], [0, 0, 1, -100, -100]])
    if use_crf:
        safe = torch.where(labels == -100, torch.zeros_like(labels), labels)
        loss = -m.crf(emissions.float(), safe, mask=mask, reduction="mean")
        dec = m.crf.decode(emissions, mask=mask)
        assert len(dec) == 2 and len(dec[0]) == 5
    else:
        loss = m.loss_fct(emissions.reshape(-1, 3), labels.reshape(-1))
    assert torch.isfinite(loss), f"non-finite loss (use_crf={use_crf})"
ok("TokenClassifier heads produce finite loss and well-formed decodes")

# --- BiLSTM-CRF end to end --------------------------------------------------
X = [["beli", "bitcoin", "sekarang"], ["harga", "eth", "naik"], ["tidak", "ada", "apa"]] * 12
Y = [["O", "B-CRYPTO", "O"], ["O", "B-CRYPTO", "O"], ["O", "O", "O"]] * 12
model, predict = U.train_bilstm_crf(X, Y, epochs=25, device="cpu", batch_size=8)
pred = predict(X[:3])
assert all(len(p) == len(x) for p, x in zip(pred, X[:3]))
f1 = U.entity_f1(Y[:3], pred)["f1"]
assert f1 > 0.5, f"BiLSTM-CRF failed to fit a trivial set (F1={f1})"
ok(f"train_bilstm_crf fits toy data (F1={f1:.3f}) and returns aligned predictions")

print("\nAll checks passed.")
