# Data

The data is not stored in this repository. Download it from Mendeley Data:
[doi:10.17632/rtw4d5c6hj](https://doi.org/10.17632/rtw4d5c6hj).

| File | Content |
|---|---|
| `gold_standard_bio_labeled.csv` | Gold Standard: 1,461 tweets, 50,242 tokens, 1,182 entities, manual BIO labels |
| `silver_standard_bio.csv` | Silver Standard: 25,726 tweets, 850,987 tokens, 37,765 entities, distant supervision with heuristic denoising |
| `lkb_cleaned.json` | LKB, Unambiguous tier (1,533 keys, 1,672 surface forms) |
| `lkb_case_sensitive.json` | LKB, Ambiguous tier (171 keys, 286 surface forms) |

Each CSV has three columns: `Sentence_ID`, `Token`, `Label`. One row is one
token. The labels are `O`, `B-CRYPTO` and `I-CRYPTO`. Mentions and URLs are
replaced with `@USER` and `HTTPURL`.

Put the files in this folder, or in `MyDrive/crypto-ner/data/` on Colab.
The notebooks expect these exact file names.

## Collection queries

The raw corpus (60,283 tweets) came from two Twitter (X) advanced search
queries:

```
(bitcoin OR btc OR ethereum OR eth OR solana OR sol OR bnb OR usdt OR xrp OR usdc OR tron OR trx OR hype OR hyperliquid) AND (beli OR jual OR serok OR harga OR cuan OR nyangkut OR terbang OR nyungsep OR borong OR rungkad OR terjun OR junam OR haka OR hajar OR koreksi OR cicil) since:2025-04-01 until:2026-04-01
```

```
(kripto OR crypto OR altcoin OR koin OR coin OR token OR memecoin OR gamefi OR rwa OR privacy OR depin OR airdrop OR staking) AND (beli OR jual OR serok OR harga OR cuan OR nyangkut OR terbang OR nyungsep OR borong OR rungkad OR terjun OR junam OR haka OR hajar OR koreksi OR cicil) since:2025-04-01 until:2026-04-01
```
