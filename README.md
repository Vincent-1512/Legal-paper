# ELECTRA PAP_NER experiments (V1–V4)

Versioned notebooks, runners, configurations, evaluation outputs, predictions, reports, and plots for the four ELECTRA runs. The original source folder remains in `electra/`; this repository is a clean, version-oriented copy.

## Run summary

| Version | Configuration | Encoder | Data | Strict entity micro-F1 | Notes |
|---|---|---|---|---:|---|
| V1 | CRF | `NlpHUST/electra-base-vn` | Original PAP_NER, test 10,278 | 97.70% | Seed 42; best checkpoint selected by strict entity micro-F1. |
| V2 | Softmax | `FPTAI/velectra-base-discriminator-cased` | Original PAP_NER, test 10,278 | 97.24% | Uses a different ELECTRA checkpoint and early-stop patience from V1, so V1–V2 is not a controlled decoder-only comparison. |
| V3 | CRF | `NlpHUST/electra-base-vn` | Deduplicated PAP_NER, test 5,162 | 98.44% | CRF reduction is `sum`. |
| V4 | CRF + Hierarchical SupCon | `NlpHUST/electra-base-vn` | Deduplicated PAP_NER, test 5,162 | 98.85% | CRF reduction is `mean`; SupCon is enabled. No CRF-mean-only control is included, so the difference from V3 cannot be attributed to SupCon alone. |

Scores are read from the saved `test_metrics.json` files. They are single-run results (seed 42). Original and deduplicated protocols use different test sets and must be reported separately.

## Data revisions

- Original PAP_NER: `Leekien0108/PAP_NER`, revision `7a0f6b1233667702925baa69c07c55d5d26eb6cd` (train/dev/test: 142,218 / 10,305 / 10,278 sentences). V1 uses local files whose SHA-256 matches the original split; V2 records the Hugging Face revision directly.
- Deduplicated PAP_NER: `levandai1704/PAP_NER_dedup_70_15_15`, revision `e4af303d22f878e5827c94861f50f747a6b3ed45` (train/dev/test: 24,089 / 5,162 / 5,162 sentences).

The notebooks retrieve the dataset from Hugging Face. Raw split files are not duplicated in this repository. Read each version's run configuration before reproducing it.

## Repository contents and large files

Each version directory contains its notebook, run configuration, metrics, saved predictions, evaluation reports, plots, tokenizer/config files, and runner where available. GitHub rejects ordinary Git files larger than 100 MiB. Model checkpoints and the `*-results.zip` bundles exceed that limit, so they are intentionally omitted here; see [`LARGE_ARTIFACTS_EXCLUDED.md`](LARGE_ARTIFACTS_EXCLUDED.md) for their names and local sizes. Raw dataset files are also omitted because the notebooks load pinned Hugging Face revisions. The original files remain in the user's `electra/` folder.

## Method attribution

The V4 hierarchical supervised contrastive setup was adapted for ELECTRA with reference to [`nghiemmmm/PhoBERT-Legal-NER`](https://github.com/nghiemmmm/PhoBERT-Legal-NER). The four experiment results and their limitations are documented above; cite the reference implementation when describing the method.

## Version directories

- [`version-1-crf-original/`](version-1-crf-original/)
- [`version-2-softmax-original/`](version-2-softmax-original/)
- [`version-3-crf-dedup/`](version-3-crf-dedup/)
- [`version-4-crf-hsupcon-dedup/`](version-4-crf-hsupcon-dedup/)
