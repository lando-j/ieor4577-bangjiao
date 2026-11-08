# ieor4577-bangjiao
Bang Jiao's work on IEOR 4577

## Movie recommendation harness

Course task: rank all 20,720 movies for 10,000 dev users; metric Recall@20 on the held-out 20%
(full catalog, no sampled negatives). Data files are provided by the course and are **not** in this repo.

## Reproduce the popularity baseline (Recall@20 = 0.1472)

```bash
cd <folder with interactions.csv, dev_labels.csv>
python3 <this repo>/score.py
# users scored: 10000 / Recall@20: 0.1472 / Precision@20: 0.0796
```

`score.py` is the official course scorer (unchanged). Score any model with `python3 score.py preds.csv`.

## Setup for the other scripts

```bash
python3 -m venv venv && venv/bin/pip install numpy scipy pandas torch
export IEOR_DATA=<folder with the csv files>      # default: the author's local path
```

## Files

| file | what |
|---|---|
| `common.py` | load data into a sparse user×item matrix, top-k, Recall@20, write predictions |
| `log.py` | appends one row per experiment to `RESULTS.md` |
| `knn.py`, `ease.py` | Item-KNN (cosine) and EASE (closed-form linear autoencoder) |
| `diag.py` | diagnostics: recall by history length / item popularity, date structure |
| `exp_debias.py`, `exp_time_alpha.py` | popularity-discount and release-year experiments |
| `vae.py` | Mult-VAE / Mult-DAE (PyTorch); model selection on 5,000 held-out training users |
| `RESULTS.md` | every run, with parameters and Recall@20 |
| `LOG.md`, `NOTES.md`, `PLAN.md` | chronological log, findings, plan |

## Results so far (dev, Recall@20)

| model | Recall@20 |
|---|---|
| Popularity (top-100) | 0.1472 |
| Item-KNN | 0.2566 |
| EASE (λ=300) | 0.3551 |
| EASE + popularity discount + release-year filter | 0.3577 |
