# V6 - Flat SupCon + Linear Probe on deduplicated PAP_NER

Baseline without hierarchy (no alpha) and without joint training (no lambda):
- Stage 1: pure Supervised Contrastive loss on entity tokens (temperature 0.07), encoder fine-tuned, early stop on dev SupCon loss (patience 5, max 15 epochs).
- Stage 2: encoder frozen, `Linear(768, 11)` trained with cross-entropy (Adam lr 1e-3, patience 5, max 20 epochs), best epoch by dev Strict F1. No CRF.
- Same dataset and revision as V5. Each seed re-runs both stages; seeds 42-46, reported as mean +- std (sample std).

Each `<MODEL>/` folder holds the self-contained Vast.ai notebook and, when run, `flatsubcon_multiseed_results/` (per-seed config, history, metrics, predictions, logs, tables).

| Model | Entity Strict micro-F1 (5 seeds) | BIO macro-F1 | Status |
|---|---:|---:|---|
| PhoBERT-base | 89.81 +- 0.31 | 94.70 +- 0.28 | done |
| BamiBERT | 86.79 +- 0.38 | 93.43 +- 0.20 | done |
| XLM-R base | - | - | notebook only, not run yet |
| ELECTRA | - | - | notebook only, not run yet |

Notes
- `source-original/` is the original repository code (unchanged). In the notebooks, `losses.py` differs from it in one place: autocast is disabled around the similarity matmul to avoid fp16 overflow (formula unchanged). `processor.py` was also changed to read CoNLL data without merging sentences and to keep tokens such as `NA`/`nan`.
- The `Strict P/R/F1` lines printed in the training logs come from the legacy CoNLL scorer and are exactly half of the real value (gold and predicted phrases are counted twice). The reported numbers in the tables are the nervaluate Strict scores.
- Runs use fp16 on CUDA, so seeds are not bit-identical.
- `tai-lieu-sheet/` contains helper prompts and the supplementary metric table (BIO macro precision, entity micro/macro P/R/F1).
- Checkpoints (`best_*.pt`) and zip bundles are not included.
