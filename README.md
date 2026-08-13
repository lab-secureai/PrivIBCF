# PrivIBCF experiments

This repository contains the experimental code used to evaluate PrivIBCF on
MovieLens-1M, Netflix Prize, and Amazon Book.

The code covers:

- dataset preparation and nested scalability subsets;
- Adjusted Cosine similarity and Weighted Sum prediction;
- early elimination of non-positive item pairs;
- communication and operation-count analysis;
- RMSE/MAE and Top-N evaluation;
- fixed-point sensitivity and recommendation agreement;
- sparsity experiments;
- local Phase-2 prediction time;
- small-scale secure multi-sum checks.

## Colab

Open `notebooks/PrivIBCF_experiments_colab.ipynb` after cloning the repository.
The notebook is intentionally short; the implementation is kept in
`src/privibcf_experiments.py`.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py
```

## Data

MovieLens-1M and Amazon Book are downloaded by the experiment code. Netflix
Prize is obtained through KaggleHub and may require Kaggle authentication.

The default subsets are:

| Dataset | Users | Item settings |
|---|---:|---|
| MovieLens-1M | 3,000 | 500, 1,000, 1,500, 2,000 |
| Netflix Prize | 5,000 | 500, 1,000, 1,500, 2,000 |
| Amazon Book | 10,000 | 200, 400, 600, 800, 1,000 |

## Reproducibility notes

Large-scale cryptographic timing is estimated from protocol operation counts
and measured modular-arithmetic costs. These estimates do not include bounded
discrete-logarithm recovery. Small-scale secure multi-sum checks are executed
directly.

The Van/Dung Phase-1 runtime values included in the code are the values reported
in the manuscript and are stored separately from measurements produced by this
implementation.

Experiment outputs are written under `results/`.
