"""
utils.py — Shared utilities for the crypto-NER revision experiments.

Covers every experiment reported in the revised paper:
  * IndoBERTweet-CRF (proposed) and all ablation variants
  * Transformer baselines (IndoBERT, mBERT, ...) and non-neural baselines
  * Reproducible 5-fold splits shared across every system
  * Out-of-fold (OOF) prediction dumping -> significance tests + error analysis

Design notes
------------
1. NO test-set leakage. Early stopping / best-checkpoint selection on the test
   fold is removed. Every configuration trains for a FIXED number of epochs so
   that ablation deltas measure the component, not the training budget.
2. Predictions are always returned at WORD level with len == len(words).
   Words lost to `max_length` truncation are padded with 'O', so truncated gold
   entities are correctly counted as false negatives instead of being silently
   dropped from both sides of the comparison.
3. One fold assignment (KFold(5, shuffle=True, random_state=42)) is generated
   once and reused by every system, which is what makes paired significance
   testing valid.

Requires: torch, transformers, datasets, pytorch-crf, seqeval, scikit-learn
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LABEL_LIST = ["O", "B-CRYPTO", "I-CRYPTO"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_LIST)

MAX_LENGTH = 128
N_SPLITS = 5
FOLD_SEED = 42

# Same tokenizer regex used to build the BIO files. Do not change it: the gold
# CSV was annotated against exactly this tokenisation.
TOKEN_PATTERN = r"@USER|HTTPURL|\#\w+|\$\w+|[a-zA-Z0-9_]+|[^\w\s]"

FINANCIAL_KEYWORDS = [
    "serok", "cuan", "nyangkut", "terbang", "nyungsep", "rungkad", "haka", "cicil",
    "pump", "dump", "hold", "hodl", "longsor", "moon", "whale", "bullish", "bearish",
    "kripto", "crypto", "koin", "coin", "token", "market",
    "pasar", "naik", "turun", "profit", "loss", "akumulasi", "wallet", "dompet", "swap",
]

BLUE_CHIPS = {"btc", "bitcoin", "eth", "ethereum", "sol", "solana", "bnb", "xrp", "ripple"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_bio_csv(path, token_col="Token", label_col="Label", id_col="Sentence_ID"):
    """Read a BIO CSV into parallel lists of (sentence_ids, sentences, labels)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=[token_col, label_col])
    df = df[df[token_col].astype(str).str.strip() != ""]

    ids, sentences, labels = [], [], []
    for sid, group in df.groupby(id_col, sort=False):
        toks = group[token_col].astype(str).tolist()
        tags = [t if t in LABEL_TO_ID else "O" for t in group[label_col].astype(str).tolist()]
        if len(toks) == 0:
            continue
        ids.append(str(sid))
        sentences.append(toks)
        labels.append(tags)
    return ids, sentences, labels


def load_lkb(filepath):
    """Load a Local Knowledge Base JSON into a lowercase surface-form set."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    assets = set()
    for key, info in data.items():
        assets.add(key.lower().strip())
        if isinstance(info, dict):
            ticker = info.get("symbol", "")
            if ticker:
                assets.add(ticker.lower().strip())
            official = info.get("official_name", "")
            if official:
                assets.add(official.lower().strip())
                cleaned = re.sub(r"\s*\(.*?\)\s*", "", official).strip()
                if cleaned:
                    assets.add(cleaned.lower())
            for alias in info.get("aliases", []):
                if alias:
                    assets.add(alias.lower().strip())
    assets.discard("")
    return assets


def get_folds(n_samples, n_splits=N_SPLITS, seed=FOLD_SEED):
    """Deterministic fold assignment shared by EVERY system.

    Returns a list of (train_idx, test_idx) and an array mapping each sample to
    the fold in which it is held out.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(n_samples)))
    fold_of = np.empty(n_samples, dtype=int)
    for f, (_, test_idx) in enumerate(splits):
        fold_of[test_idx] = f
    return splits, fold_of


def inner_dev_split(train_idx, dev_ratio=0.0, seed=42):
    """Optionally carve a dev set out of the TRAINING folds only.

    Kept available for anyone who wants early stopping without leakage, but the
    default protocol uses a fixed epoch budget and no dev set at all.
    """
    if dev_ratio <= 0:
        return train_idx, np.array([], dtype=int)
    rng = np.random.RandomState(seed)
    idx = np.array(train_idx)
    rng.shuffle(idx)
    n_dev = max(1, int(len(idx) * dev_ratio))
    return idx[n_dev:], idx[:n_dev]


# --------------------------------------------------------------------------- #
# Entity-level metrics (seqeval-compatible, dependency-free)
# --------------------------------------------------------------------------- #

def extract_entities(labels):
    """BIO tag sequence -> list of (start, end_exclusive, type) spans."""
    spans, start, etype = [], None, None
    for i, lab in enumerate(labels):
        if lab.startswith("B-"):
            if start is not None:
                spans.append((start, i, etype))
            start, etype = i, lab[2:]
        elif lab.startswith("I-"):
            if start is None:
                start, etype = i, lab[2:]      # tolerate stray I- (softmax variant)
            elif lab[2:] != etype:
                spans.append((start, i, etype))
                start, etype = i, lab[2:]
        else:
            if start is not None:
                spans.append((start, i, etype))
                start, etype = None, None
    if start is not None:
        spans.append((start, len(labels), etype))
    return spans


def entity_counts(y_true, y_pred):
    """Aggregate TP / n_pred / n_true over a corpus, entity-level exact match."""
    tp = n_pred = n_true = 0
    for t, p in zip(y_true, y_pred):
        gold = set(extract_entities(t))
        pred = set(extract_entities(p))
        tp += len(gold & pred)
        n_pred += len(pred)
        n_true += len(gold)
    return tp, n_pred, n_true


def prf(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def entity_f1(y_true, y_pred):
    """Micro precision / recall / F1 under the exact-match criterion."""
    p, r, f = prf(*entity_counts(y_true, y_pred))
    return {"precision": p, "recall": r, "f1": f}


def count_bio_violations(y_pred):
    """Structurally invalid transitions: I-X not preceded by B-X or I-X."""
    bad = total = 0
    for seq in y_pred:
        prev = "O"
        for lab in seq:
            total += 1
            if lab.startswith("I-"):
                if not (prev.startswith("B-") and prev[2:] == lab[2:]) and \
                   not (prev.startswith("I-") and prev[2:] == lab[2:]):
                    bad += 1
            prev = lab
    return bad, total


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class TokenClassifier(nn.Module):
    """Transformer encoder + CRF head, or + softmax head for the -CRF ablation."""

    def __init__(self, model_checkpoint, num_labels=NUM_LABELS, use_crf=True, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.use_crf = use_crf
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.bert = AutoModel.from_pretrained(model_checkpoint, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_tags=num_labels, batch_first=True)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)
        emissions = self.classifier(seq)
        mask = attention_mask.bool()

        if self.use_crf:
            decoded = self.crf.decode(emissions, mask=mask)
            if labels is None:
                return {"logits": decoded}
            safe = torch.where(labels == -100, torch.zeros_like(labels), labels)
            # float32 for the CRF partition function: fp16 log-sum-exp is unstable
            loss = -self.crf(emissions.float(), safe, mask=mask, reduction="mean")
            max_len = emissions.shape[1]
            padded = [p + [0] * (max_len - len(p)) for p in decoded]
            return {"loss": loss, "logits": torch.tensor(padded, device=emissions.device)}

        decoded = emissions.argmax(dim=-1)
        if labels is None:
            return {"logits": [d.tolist() for d in decoded]}
        loss = self.loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": decoded}


def load_state_dict_any(path):
    """Load weights from either safetensors or a .bin checkpoint directory."""
    st = os.path.join(path, "model.safetensors")
    bin_ = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    if os.path.exists(bin_):
        return torch.load(bin_, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {path}")


# --------------------------------------------------------------------------- #
# Tokenisation & inference
# --------------------------------------------------------------------------- #

def tokenize_and_align(tokenizer, sentences, tag_seqs, max_length=MAX_LENGTH):
    enc = tokenizer(
        list(sentences),
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    aligned = []
    for i, tags in enumerate(tag_seqs):
        word_ids = enc.word_ids(batch_index=i)
        prev, ids = None, []
        for wid in word_ids:
            if wid is None:
                ids.append(-100)
            elif wid != prev:
                ids.append(LABEL_TO_ID.get(tags[wid], 0))
            else:
                ids.append(-100)
            prev = wid
        aligned.append(ids)
    enc["labels"] = aligned
    return enc


@torch.no_grad()
def predict_words(model, tokenizer, sentences, device, max_length=MAX_LENGTH, batch_size=32):
    """Word-level predictions. Always returns len(pred) == len(words) per sentence.

    Words dropped by truncation are filled with 'O' rather than removed, so a
    truncated gold entity counts as a false negative instead of vanishing from
    both prediction and reference.
    """
    model.eval()
    all_preds, n_truncated = [], 0

    for start in range(0, len(sentences), batch_size):
        batch = [list(s) for s in sentences[start:start + batch_size]]
        enc = tokenizer(
            batch, is_split_into_words=True, truncation=True,
            max_length=max_length, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in enc.items()}
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = out["logits"]

        for b, words in enumerate(batch):
            seq = logits[b] if not torch.is_tensor(logits) else logits[b].tolist()
            word_ids = enc.word_ids(batch_index=b)
            labels = ["O"] * len(words)
            prev, covered = None, 0
            for j, wid in enumerate(word_ids):
                if wid is None or wid == prev:
                    prev = wid
                    continue
                if j < len(seq):
                    labels[wid] = ID_TO_LABEL[int(seq[j])]
                    covered = max(covered, wid + 1)
                prev = wid
            if covered < len(words):
                n_truncated += 1
            all_preds.append(labels)

    if n_truncated:
        print(f"    [note] {n_truncated} sentence(s) truncated at max_length={max_length}; "
              f"unreached words padded with 'O'.")
    return all_preds


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _make_trainer(model, tokenizer, train_ds, args):
    from transformers import DataCollatorForTokenClassification, Trainer

    class CRFTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    return CRFTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )


def _training_args(output_dir, lr, batch_size, epochs, fp16, seed=42):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="no",           # no checkpoint selection at all
        # NOTE: evaluation during training is deliberately omitted. No
        # eval_dataset is ever passed to the Trainer, so the test fold cannot
        # influence model selection. The argument name for this changed between
        # transformers versions (evaluation_strategy -> eval_strategy), so it is
        # left unset rather than pinned: "no" is the default either way.
        logging_steps=100,
        report_to=[],
        fp16=fp16,
        seed=seed,
    )


def train_base_model(silver_bio_path, encoder, output_dir, use_crf=True,
                     lr=5e-5, batch_size=32, epochs=3, dev_ratio=0.1, seed=42):
    """Stage 1: train on the Silver Standard produced by distant supervision.

    The silver set is disjoint from the gold set, so a single base checkpoint is
    valid for all five folds. A held-out slice of SILVER (never gold) is used
    only to report convergence.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    set_seed(seed)
    device = get_device()
    _, sentences, labels = load_bio_csv(silver_bio_path)
    print(f"Silver sentences: {len(sentences)}")

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    enc = tokenize_and_align(tokenizer, sentences, labels)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["labels"],
    })
    split = ds.train_test_split(test_size=dev_ratio, seed=seed)

    model = TokenClassifier(encoder, use_crf=use_crf).to(device)
    args = _training_args(output_dir, lr, batch_size, epochs, torch.cuda.is_available())
    trainer = _make_trainer(model, tokenizer, split["train"], args)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    print(f"Base model saved -> {output_dir}")
    return output_dir


def run_experiment(config, sentences, labels, sentence_ids, splits, out_dir="oof"):
    """Run one system across all folds and dump out-of-fold predictions.

    config keys
    -----------
    name        str   identifier, also the OOF filename
    type        str   'transformer' | 'pretrained_only' | 'gazetteer' | 'bilstm'
    encoder     str   HF checkpoint (transformer types)
    use_crf     bool  CRF head vs softmax head
    init_from   str   base-checkpoint dir to initialise from, or None for gold-only
    lr, batch_size, epochs
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    name = config["name"]
    ctype = config.get("type", "transformer")
    device = get_device()
    print(f"\n=== {name} ({ctype}) ===")

    oof_pred = [None] * len(sentences)
    fold_of = np.empty(len(sentences), dtype=int)

    # ---- systems that never see gold at training time: predict once ---------
    if ctype in ("gazetteer", "pretrained_only"):
        if ctype == "gazetteer":
            preds = gazetteer_predict(sentences, config["lkb_unambiguous"],
                                      config["lkb_ambiguous"], config.get("denoise", True))
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["init_from"])
            model = TokenClassifier(config["encoder"], use_crf=config.get("use_crf", True))
            model.load_state_dict(load_state_dict_any(config["init_from"]))
            model.to(device)
            preds = predict_words(model, tokenizer, sentences, device)
            del model
            torch.cuda.empty_cache()
        for f, (_, test_idx) in enumerate(splits):
            for i in test_idx:
                oof_pred[i] = preds[i]
                fold_of[i] = f

    # ---- systems trained per fold -----------------------------------------
    else:
        for f, (train_idx, test_idx) in enumerate(splits):
            print(f"  fold {f + 1}/{len(splits)}")
            set_seed(config.get("seed", 42) + f)
            X_tr = [sentences[i] for i in train_idx]
            y_tr = [labels[i] for i in train_idx]
            X_te = [sentences[i] for i in test_idx]

            if ctype == "bilstm":
                model, predict_fn = train_bilstm_crf(
                    X_tr, y_tr, epochs=config.get("epochs", 20),
                    lr=config.get("lr", 1e-3), use_crf=config.get("use_crf", True),
                    device=device)
                preds = predict_fn(X_te)
            else:
                init = config.get("init_from")
                tok_src = init if init else config["encoder"]
                tokenizer = AutoTokenizer.from_pretrained(tok_src)
                model = TokenClassifier(config["encoder"], use_crf=config.get("use_crf", True))
                if init:
                    model.load_state_dict(load_state_dict_any(init))
                model.to(device)

                enc = tokenize_and_align(tokenizer, X_tr, y_tr)
                ds = Dataset.from_dict({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": enc["labels"],
                })
                args = _training_args(f"/tmp/{name}_f{f}", config["lr"],
                                      config["batch_size"], config["epochs"],
                                      torch.cuda.is_available(),
                                      seed=config.get("seed", 42) + f)
                _make_trainer(model, tokenizer, ds, args).train()
                preds = predict_words(model, tokenizer, X_te, device)

            for k, i in enumerate(test_idx):
                oof_pred[i] = preds[k]
                fold_of[i] = f
            del model
            torch.cuda.empty_cache()

    # ---- dump + summarise --------------------------------------------------
    path = os.path.join(out_dir, f"oof_{name}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for i, (sid, toks, gold, pred) in enumerate(
                zip(sentence_ids, sentences, labels, oof_pred)):
            fh.write(json.dumps({
                "sentence_id": sid, "index": i, "fold": int(fold_of[i]),
                "tokens": toks, "true": gold, "pred": pred,
            }, ensure_ascii=False) + "\n")

    per_fold = []
    for f in range(len(splits)):
        idx = [i for i in range(len(sentences)) if fold_of[i] == f]
        m = entity_f1([labels[i] for i in idx], [oof_pred[i] for i in idx])
        per_fold.append(m)
    pooled = entity_f1(labels, oof_pred)

    print(f"  pooled  P={pooled['precision']:.4f} R={pooled['recall']:.4f} F1={pooled['f1']:.4f}")
    print(f"  per-fold F1 mean={np.mean([m['f1'] for m in per_fold]):.4f} "
          f"(± {np.std([m['f1'] for m in per_fold]):.4f})  -> {path}")
    return {"name": name, "per_fold": per_fold, "pooled": pooled, "oof_path": path}


# --------------------------------------------------------------------------- #
# Baseline 1: gazetteer / longest match (no training)
# --------------------------------------------------------------------------- #

def gazetteer_label_tokens(tokens, lkb_unambiguous, lkb_ambiguous,
                           financial_keywords=FINANCIAL_KEYWORDS, denoise=True,
                           max_ngram=4):
    """Longest-match distant supervision over pre-tokenised text.

    denoise=True  -> the Heuristic-Augmented Denoising rules of the paper
    denoise=False -> naive matching, used to build the -HD ablation silver set
    """
    labels = ["O"] * len(tokens)
    i = 0
    while i < len(tokens):
        ori = tokens[i]
        core = re.sub(r"^[\$\#]", "", ori)
        core_lower = core.lower()

        if ori[:1] in ("$", "#") and len(ori) > 1:
            if not re.match(r"^[\d\.,]+[kKmMbB]?$", core):
                if core_lower in lkb_unambiguous or core_lower in lkb_ambiguous:
                    labels[i] = "B-CRYPTO"
                i += 1
                continue

        matched = False
        for window in range(max_ngram, 0, -1):
            if i + window > len(tokens):
                continue
            phrase = " ".join(tokens[i:i + window]).lower()

            if phrase in lkb_unambiguous:
                labels[i] = "B-CRYPTO"
                for j in range(1, window):
                    labels[i + j] = "I-CRYPTO"
                i += window
                matched = True
                break

            if window == 1 and phrase in lkb_ambiguous:
                if not denoise:
                    labels[i] = "B-CRYPTO"
                    i += 1
                    matched = True
                    break
                if ori.isupper():
                    lo, hi = max(0, i - 4), min(len(tokens), i + 5)
                    ctx = [t.lower() for t in tokens[lo:hi]]
                    if any(kw in ctx for kw in financial_keywords):
                        labels[i] = "B-CRYPTO"
                        i += 1
                        matched = True
                        break
        if not matched:
            i += 1
    return labels


def gazetteer_predict(sentences, lkb_unambiguous, lkb_ambiguous, denoise=True):
    return [gazetteer_label_tokens(s, lkb_unambiguous, lkb_ambiguous, denoise=denoise)
            for s in sentences]


# --------------------------------------------------------------------------- #
# Baseline 2: BiLSTM-CRF (non-transformer neural baseline)
# --------------------------------------------------------------------------- #

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_labels=NUM_LABELS, emb_dim=100,
                 hidden=200, use_crf=True, pad_idx=0):
        super().__init__()
        self.use_crf = use_crf
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, num_labels)
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_tags=num_labels, batch_first=True)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, mask, labels=None):
        h, _ = self.lstm(self.emb(x))
        emissions = self.fc(self.dropout(h))
        if self.use_crf:
            decoded = self.crf.decode(emissions, mask=mask)
            if labels is None:
                return decoded
            safe = torch.where(labels < 0, torch.zeros_like(labels), labels)
            return -self.crf(emissions, safe, mask=mask, reduction="mean")
        if labels is None:
            return [row[m].tolist() for row, m in zip(emissions.argmax(-1), mask)]
        return self.loss_fct(emissions.reshape(-1, emissions.size(-1)), labels.reshape(-1))


def train_bilstm_crf(X_train, y_train, epochs=20, lr=1e-3, emb_dim=100,
                     hidden=200, use_crf=True, batch_size=32, device="cpu",
                     pretrained_vectors=None):
    """Train a BiLSTM-CRF from scratch and return (model, predict_fn).

    `pretrained_vectors` accepts a dict {word: np.array} (e.g. FastText id) to
    initialise the embedding table; leave None for random initialisation.
    """
    vocab = {"<pad>": 0, "<unk>": 1}
    for sent in X_train:
        for w in sent:
            wl = w.lower()
            if wl not in vocab:
                vocab[wl] = len(vocab)

    def encode(sent):
        return [vocab.get(w.lower(), 1) for w in sent]

    model = BiLSTMCRF(len(vocab), emb_dim=emb_dim, hidden=hidden, use_crf=use_crf).to(device)

    if pretrained_vectors:
        with torch.no_grad():
            hits = 0
            for w, i in vocab.items():
                if w in pretrained_vectors:
                    model.emb.weight[i] = torch.tensor(pretrained_vectors[w][:emb_dim])
                    hits += 1
            print(f"    embedding init: {hits}/{len(vocab)} words matched")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = list(range(len(X_train)))

    def make_batch(sents, tags=None):
        maxlen = max(len(s) for s in sents)
        x = torch.zeros(len(sents), maxlen, dtype=torch.long)
        mask = torch.zeros(len(sents), maxlen, dtype=torch.bool)
        y = torch.full((len(sents), maxlen), -100, dtype=torch.long)
        for i, s in enumerate(sents):
            ids = encode(s)
            x[i, :len(ids)] = torch.tensor(ids)
            mask[i, :len(ids)] = True
            if tags is not None:
                y[i, :len(ids)] = torch.tensor([LABEL_TO_ID.get(t, 0) for t in tags[i]])
        return x.to(device), mask.to(device), y.to(device)

    model.train()
    for ep in range(epochs):
        random.shuffle(order)
        total = 0.0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            x, mask, y = make_batch([X_train[i] for i in idx], [y_train[i] for i in idx])
            loss = model(x, mask, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()

    @torch.no_grad()
    def predict_fn(X):
        model.eval()
        out = []
        for s in range(0, len(X), batch_size):
            chunk = X[s:s + batch_size]
            x, mask, _ = make_batch(chunk)
            decoded = model(x, mask)
            for words, seq in zip(chunk, decoded):
                labs = [ID_TO_LABEL[int(t)] for t in seq][:len(words)]
                labs += ["O"] * (len(words) - len(labs))
                out.append(labs)
        return out

    return model, predict_fn


# --------------------------------------------------------------------------- #
# Significance testing
# --------------------------------------------------------------------------- #

def load_oof(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["index"])
    return rows


def per_sentence_counts(y_true, y_pred):
    """Per-sentence (tp, n_pred, n_true) arrays — computed ONCE per system.

    Resampling tests only ever need these three numbers per sentence, so entity
    extraction runs len(y_true) times instead of once per resample. Without
    this, 2,000 resamples over 1,461 sentences means millions of redundant
    extractions and the tests take hours rather than seconds.
    """
    n = len(y_true)
    tp = np.empty(n, dtype=np.int64)
    n_pred = np.empty(n, dtype=np.int64)
    n_true = np.empty(n, dtype=np.int64)
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        gold = set(extract_entities(t))
        pred = set(extract_entities(p))
        tp[i] = len(gold & pred)
        n_pred[i] = len(pred)
        n_true[i] = len(gold)
    return tp, n_pred, n_true


def _f1_from_sums(tp, n_pred, n_true):
    """Micro-F1 from aggregate counts. Works on scalars or arrays of resamples."""
    tp = np.asarray(tp, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    p = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    r = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    denom = p + r
    return np.divide(2 * p * r, denom, out=np.zeros_like(tp), where=denom > 0)


def _resample_f1(tp, n_pred, n_true, idx):
    return _f1_from_sums(tp[idx].sum(1), n_pred[idx].sum(1), n_true[idx].sum(1))


def paired_bootstrap(y_true, pred_a, pred_b, n_resamples=10000, seed=42, block=500):
    """Two-sided paired bootstrap on entity-level micro-F1 (system B vs A).

    Resamples SENTENCES with replacement, which is the correct unit here because
    both systems predict on exactly the same sentences. Resampling is done in
    blocks so peak memory stays a few MB regardless of n_resamples.
    """
    tp_a, np_a, nt = per_sentence_counts(y_true, pred_a)
    tp_b, np_b, _ = per_sentence_counts(y_true, pred_b)

    observed = float(_f1_from_sums(tp_b.sum(), np_b.sum(), nt.sum()) -
                     _f1_from_sums(tp_a.sum(), np_a.sum(), nt.sum()))

    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples)
    for s in range(0, n_resamples, block):
        b = min(block, n_resamples - s)
        idx = rng.randint(0, n, size=(b, n))
        deltas[s:s + b] = (_resample_f1(tp_b, np_b, nt, idx) -
                           _resample_f1(tp_a, np_a, nt, idx))

    centred = deltas - deltas.mean()
    p = float((np.abs(centred) >= abs(observed)).mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_f1": observed, "p_value": p, "ci_low": float(lo), "ci_high": float(hi)}


def approximate_randomization(y_true, pred_a, pred_b, n_trials=10000, seed=42, block=500):
    """Two-sided paired approximate randomization test (Yeh, 2000).

    Under the null the two systems are interchangeable, so each sentence's
    prediction pair is swapped with probability 0.5 and the F1 gap recomputed.
    """
    tp_a, np_a, nt = per_sentence_counts(y_true, pred_a)
    tp_b, np_b, _ = per_sentence_counts(y_true, pred_b)

    observed = abs(float(_f1_from_sums(tp_b.sum(), np_b.sum(), nt.sum()) -
                         _f1_from_sums(tp_a.sum(), np_a.sum(), nt.sum())))

    rng = np.random.RandomState(seed)
    n = len(y_true)
    count = 0
    for s in range(0, n_trials, block):
        b = min(block, n_trials - s)
        swap = rng.rand(b, n) < 0.5
        tp_x = np.where(swap, tp_b, tp_a).sum(1)
        np_x = np.where(swap, np_b, np_a).sum(1)
        tp_y = np.where(swap, tp_a, tp_b).sum(1)
        np_y = np.where(swap, np_a, np_b).sum(1)
        nt_sum = nt.sum()
        gap = np.abs(_f1_from_sums(tp_y, np_y, nt_sum) -
                     _f1_from_sums(tp_x, np_x, nt_sum))
        count += int((gap >= observed).sum())
    return {"p_value": (count + 1) / (n_trials + 1)}


def bootstrap_ci(y_true, y_pred, n_resamples=10000, seed=42, block=500):
    """Percentile bootstrap 95% CI for entity-level micro-F1."""
    tp, n_pred, n_true = per_sentence_counts(y_true, y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = np.empty(n_resamples)
    for s in range(0, n_resamples, block):
        b = min(block, n_resamples - s)
        idx = rng.randint(0, n, size=(b, n))
        scores[s:s + b] = _resample_f1(tp, n_pred, n_true, idx)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def holm_bonferroni(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down correction. Returns adjusted p and reject flags."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvalues[i]
        running = max(running, min(val, 1.0))
        adj[i] = running
    return adj, adj < alpha


# --------------------------------------------------------------------------- #
# Inter-annotator agreement
# --------------------------------------------------------------------------- #

def fleiss_kappa(table):
    """Fleiss' kappa. `table` is (n_items, n_categories) of rater counts."""
    table = np.asarray(table, dtype=float)
    n_items, _ = table.shape
    n_raters = table.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item")
    n = n_raters[0]
    p_j = table.sum(axis=0) / (n_items * n)
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) else 1.0


def token_kappa_table(annotations):
    """annotations: list of label sequences per annotator, aligned token-wise."""
    n_ann = len(annotations)
    flat = [sum(a, []) if isinstance(a[0], list) else list(a) for a in annotations]
    n_tok = len(flat[0])
    assert all(len(f) == n_tok for f in flat), "annotator token counts differ"
    table = np.zeros((n_tok, NUM_LABELS))
    for a in flat:
        for i, lab in enumerate(a):
            table[i, LABEL_TO_ID.get(lab, 0)] += 1
    assert table.sum(axis=1).min() == n_ann
    return table


def pairwise_entity_agreement(annotations):
    """Symmetric entity-level F1 between every pair of annotators."""
    out = {}
    for a in range(len(annotations)):
        for b in range(a + 1, len(annotations)):
            m = entity_f1(annotations[a], annotations[b])
            out[f"A{a + 1}-A{b + 1}"] = m["f1"]
    return out


# --------------------------------------------------------------------------- #
# Dataset descriptive statistics
# --------------------------------------------------------------------------- #

def assign_strata(sentences, lkb_unambiguous, lkb_ambiguous,
                  blue_chips=BLUE_CHIPS, financial_keywords=FINANCIAL_KEYWORDS):
    """Recover the Blue-Chip / Altcoin / Background stratum of each sentence."""
    all_crypto = set(lkb_unambiguous) | set(lkb_ambiguous)
    altcoins = all_crypto - set(blue_chips)
    strata = []
    for toks in sentences:
        low = {t.lower() for t in toks}
        text = " ".join(toks).lower()
        if low & blue_chips:
            strata.append("Blue-Chip")
        elif low & altcoins:
            strata.append("Altcoin")
        elif any(kw in text for kw in financial_keywords):
            strata.append("Background")
        else:
            strata.append("Other")
    return strata


def corpus_statistics(sentences, labels):
    n_tokens = sum(len(s) for s in sentences)
    ents = [extract_entities(l) for l in labels]
    n_ents = sum(len(e) for e in ents)
    span_len = Counter(end - start for e in ents for (start, end, _) in e)
    tag_dist = Counter(t for l in labels for t in l)
    surfaces = Counter(
        " ".join(sent[s:e]).lower()
        for sent, e_list in zip(sentences, ents) for (s, e, _) in e_list
    )
    hapax = sum(1 for v in surfaces.values() if v == 1)
    return {
        "sentences": len(sentences),
        "tokens": n_tokens,
        "avg_tokens_per_sentence": n_tokens / max(1, len(sentences)),
        "entities": n_ents,
        "entities_per_sentence": n_ents / max(1, len(sentences)),
        "sentences_without_entity": sum(1 for e in ents if not e),
        "unique_surface_forms": len(surfaces),
        "hapax_surface_forms": hapax,
        "hapax_ratio": hapax / max(1, len(surfaces)),
        "tag_distribution": dict(tag_dist),
        "span_length_distribution": dict(sorted(span_len.items())),
        "top_surface_forms": surfaces.most_common(15),
    }
