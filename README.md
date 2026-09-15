# Heuristic-Augmented Denoising in Distant Supervision for Crypto Assets Named Entity Recognition Using IndoBERTweet-CRF

📄 **Accepted at ICIMCIS 2026** (International Conference on Informatics, Multimedia, Cyber, and Information System) — IEEE Technically Sponsored

## Abstract

The rapid expansion of the digital asset market in Indonesia has fostered a highly active cryptocurrency community on Twitter (X), necessitating automated systems to monitor discussions and mitigate associated financial risks. However, extracting entities from informal social media text is challenging due to the community's complex linguistic registers, high out-of-vocabulary rates, and the critical scarcity of domain-specific annotated datasets for the Indonesian language. To address dataset scarcity, this study proposes a hybrid Named Entity Recognition (NER) pipeline utilizing an IndoBERTweet-CRF architecture. A Distant Supervision approach is employed to automate the labeling of a Silver Standard dataset using the CoinGecko API as a Local Knowledge Base, integrated with a Heuristic-Augmented Denoising mechanism to mitigate label noise arising from lexically ambiguous terms. The base model is subsequently fine-tuned on a Gold Standard corpus comprising 1,461 manually annotated tweets. Evaluated through 5-Fold Cross-Validation, the proposed model achieves a mean Precision of 77.48% and an F1-Score of 70.76%, while the Conditional Random Field (CRF) layer successfully prevents all structurally invalid label sequences.

**Keywords:** Named Entity Recognition (NER), Crypto assets, Distant supervision, IndoBERTweet-CRF, Indonesian Twitter

## 🔬 Pipeline Overview

`experiment_pipeline.ipynb` implements the full pipeline end-to-end, in five stages:

1. **Preprocessing** — normalizes raw tweets to IndoBERTweet's expected input format (mention/URL replacement, whitespace cleanup), then applies noise, language, and spam filtering
2. **Splitting (Gold/Silver)** — builds a Local Knowledge Base (LKB) from the CoinGecko API, then stratified-samples a Gold Standard subset for manual annotation while the remainder forms the Silver Standard pool
3. **Labeling** — automatically labels the Silver Standard via Distant Supervision against the LKB, with a Heuristic-Augmented Denoising step to filter out label noise from lexically ambiguous terms; Gold Standard tweets are manually annotated in BIO format
4. **Model Training:**
   - Base model pretraining on the Silver Standard (IndoBERTweet + CRF layer)
   - Base model evaluation on the Gold Standard
   - Hybrid fine-tuning with 5-Fold Cross-Validation on the Gold Standard
   - Final production fine-tuning on the full Gold Standard
   - Error analysis on model predictions

## 📊 Results

Evaluated on the Gold Standard (1,461 manually annotated tweets) via 5-Fold Cross-Validation:

| Metric | Score |
|---|---|
| Mean Precision | 77.48% |
| F1-Score | 70.76% |
| CRF structural validity | 100% (no invalid BIO transitions) |

## 🛠️ Tech Stack

- Python, pandas
- Hugging Face `transformers`, `datasets`, `evaluate`, `seqeval`
- PyTorch, `pytorch-crf`
- Google Colab (GPU training environment)

## ⚙️ Running the Notebook

This notebook was built for **Google Colab** (it mounts Google Drive and uses `/content/` paths). To reproduce:

1. Open `experiment_pipeline.ipynb` in Google Colab
2. Provide your own raw tweet dataset as `dataset_mentah_twitter.csv` (see note on data below)
3. Run the cells sequentially — each stage (Preprocessing → Splitting → Labeling → Model Training) writes the CSV that the next stage reads

## ⚠️ Note on Data

The raw tweet dataset (`dataset_mentah_twitter.csv`) was collected via web scraping from Twitter/X and is **not included in this repository**, since Twitter/X's Developer Policy generally restricts public redistribution of raw scraped tweet content. To reproduce the dataset, collect tweets using the same search queries and date range described in the paper's Methodology section (crypto-related keywords combined with Indonesian trading slang, April 2025–April 2026), then run the preprocessing pipeline on your own scrape.

## 👥 Team

- Jonathan Davin
- Clark Sompie
- Shane Anthony
- Henry Lucky
- Bryan Dale
- Rifqi Charisma

*Computer Science Department, School of Computer Science, Bina Nusantara University, Jakarta, Indonesia*

## 📚 Citation

If you use this work, please cite:

> Davin, J., Sompie, C., Anthony, S., Lucky, H., Dale, B., & Charisma, R. (2026). *Heuristic-Augmented Denoising in Distant Supervision for Crypto Assets Named Entity Recognition Using IndoBERTweet-CRF*. Accepted at the International Conference on Informatics, Multimedia, Cyber, and Information System (ICIMCIS) 2026.
