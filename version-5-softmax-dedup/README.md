# V5 - Softmax baseline on deduplicated PAP_NER (4 encoders)

Softmax token-classification baselines (no CRF, no contrastive loss) for PhoBERT-base, XLM-R base, BamiBERT and ELECTRA on the deduplicated dataset
`levandai1704/PAP_NER_dedup_70_15_15` (revision `e4af303d22f878e5827c94861f50f747a6b3ed45`; train/dev/test 24,089 / 5,162 / 5,162 sentences).

Single run per model, 20 epochs, RTX 3090. Scores are read from each run's `test_report.txt` / `advanced_eval_table.csv` (entity level, Strict).

| Model | Entity Strict micro-F1 | BIO accuracy |
|---|---:|---:|
| PhoBERT-base | 98.20 | 99.67 |
| BamiBERT | 98.15 | 99.70 |
| XLM-R base | 98.42 | 99.70 |
| ELECTRA | 98.26 | 99.70 |

Folders: `PHOBERT/`, `XMLR/`, `BAMI/`, `ELECTRA/`. Model checkpoints (`*.pt`) and result zip bundles are not included (size > 100 MiB); see the local source folders.
