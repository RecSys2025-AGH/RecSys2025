# ENSEMBLE BERT4REC FOR RECSYS 2025 CHALLENGE

This repository contains our solution developed for the **RecSys Challenge 2025**, aiming to create the best possible architecture to generate **Universal Behavioral Profiles (UBP)** of users based on their interactions with a system.  
Our solution achieved a **top 50 place out of nearly 500 participating teams**, reflecting its competitive strength in this highly challenging task.

---

## Overview

The RecSys Challenge 2025 focused on modeling user behavior across multiple types of interactions in an online system to produce universal user representations that can be effectively applied across diverse predictive tasks without task-specific tuning.  

The goal was to create embeddings capturing key behavioral patterns from user activity sequences—including purchases, cart actions, search queries, and page visits—and assess their utility in predicting future user behavior.

Our approach leverages **ensemble learning** with **BERT4Rec** models, which are transformer-based architectures featuring bidirectional self-attention, trained independently for each type of user interaction.  
By combining embeddings from each model, we produce a comprehensive user representation superior to conventional feature engineering baselines.

---

## Key Features

- **Multi-Modal User Behavior Modeling** – Separate BERT4Rec models trained on sequences of:
  - Product purchases
  - Add-to-cart and remove-from-cart events
  - Search queries
  - Page visits

- **Bidirectional Transformer Architecture** – Utilizing BERT4Rec’s context-aware bidirectional attention for improved behavior modeling.

- **Embedding Fusion Strategy** – Unifying multiple interaction-specific embeddings via concatenation.

- **Evaluation on Multiple Tasks** – Including churn prediction, product propensity, and category propensity in the challenge's neural evaluation framework.

- **Competitive Performance** – Outperformed the baseline feature aggregation method, ranking within the **top 25** on key metrics and **top 50 overall**.

---

## Data Summary

- **~170 million** events from **~19 million** anonymized users  
- Event types:
  - `product_buy` — purchases
  - `add_to_cart` & `remove_from_cart` — cart modifications
  - `page_visit` — filtered page views
  - `search_query` — compressed query embeddings

- Sparse dataset distribution — many users have minimal interactions, mimicking real-world datasets.

---

## Methodology

### Model Architecture
- **BERT4Rec** – Self-supervised sequential recommendation architecture using multi-head self-attention.
- **Separate Models per Action Type** – Trained independently due to lack of ID mapping across modalities.
- **Typical Hyperparameters**:
  - Embedding dim: 256
  - Layers: 6
  - Heads: 8
  - Dropout: 0.2
  - Max sequence length tuned per action type (30–128)

### Training & Embedding Fusion
- **Cloze-style masking** to train on predicting masked interactions.
- **Concatenated embeddings** from each BERT4Rec model form the final UBP vectors.
- Direct input of fused embeddings into the competition evaluation architecture.

### Baseline & Experiments
- Baseline: Statistical feature aggregation with temporal windows.
- Single-action models underperformed baseline.
- Multi-action ensemble surpassed all provided benchmarks.
- Failed experiments:
  - Dense layers between baseline & evaluation network
  - Autoencoder-enhanced baseline features

---

## Results

| Task                | Baseline  | BERT4Rec (Purchases) | BERT4Rec (All Actions) |
|---------------------|-----------|----------------------|------------------------|
| Churn Prediction    | 0.6947    | 0.6490               | 0.6851                 |
| Product Propensity  | 0.6985    | 0.6938               | 0.7677                 |
| Category Propensity | 0.6919    | 0.6578               | 0.7230                 |
| Hidden Test Tasks   | up to 0.7382 | up to 0.7167       | up to 0.7859           |

**Key Insight:** The ensemble across multiple behavior modalities was the main contributor to the improvement in product and category propensity predictions.

---

## Conclusion

Our solution highlights how **transformer-based self-supervised learning** can effectively model complex, multi-modal user behavior sequences.  
By training **separate BERT4Rec instances** for each interaction type and **fusing embeddings**, we achieved strong generalization across both known and hidden tasks in the RecSys Challenge 2025.

This approach paves the way for **next-generation recommender systems** by fusing power from deep sequential models and ensemble strategies.

---

## Repository Contents

- BERT4Rec training scripts for each event type.
- Data preprocessing pipelines.
- Embedding extraction & concatenation utility scripts.
- Integration with competition’s official evaluation framework.
- Configuration files with hyperparameters & experimental logs.

---

## Acknowledgements
We thank the RecSys Challenge 2025 organizers for the well-structured data and evaluation framework.

**Team Members:**  
- Adam Stajek – `adamstajek@student.agh.edu.pl`  
- Maksym Szemer – `szemermaksym@student.agh.edu.pl`  
- Adam Tokarz – `adamtokarz@student.agh.edu.pl`

AGH University of Science and Technology  
August 2025
