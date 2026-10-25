# Large ELECTRA files kept in the original local folder

These files were excluded from ordinary Git because GitHub blocks files larger than 100 MiB and the full result archives contain large checkpoints. They remain under the original `electra/` folder. Metrics, predictions, training histories, source, and reports are included in the repository. Raw split files are not duplicated; the notebooks load pinned Hugging Face revisions.

| Local path relative to `electra/` | Size |
|---|---:|
| `lần1+crf/papner_electra_crf_official_20ep_seed42_rtx3090_24gb-results.zip` | 473.2 MiB |
| `lần2_ko_crf/papner_electra_softmax_20ep_rtx3090_24gb-results.zip` | 391.7 MiB |
| `lần2_ko_crf/papner_electra_softmax_20ep_rtx3090_24gb/best_model.pt` | 420.0 MiB |
| `lần3/papner_electra_crf_dedup_70_15_15_20ep_seed42_rtx3090_24gb-results.zip` | 473.0 MiB |
| `lần3/papner_electra_crf_dedup_70_15_15_20ep_seed42_rtx3090_24gb/best_model.pt` | 507.7 MiB |
| `lần4/papner_electra_crf_hierarchical_supcon_dedup_70_15_15_20ep_seed42_rtx3090_24gb-results.zip` | 473.8 MiB |
| `lần4/papner_electra_crf_hierarchical_supcon_dedup_70_15_15_20ep_seed42_rtx3090_24gb/best_model.pt` | 508.6 MiB |
| `papner_electra_crf_official_20ep_seed42_rtx3090_24gb/best_model.pt` | 507.7 MiB |

| `PAP_NER_ELECTRA_RTX3090_24GB/Dataset/dev_data.txt` | Excluded; downloaded by notebook from the pinned dataset revision |
| `PAP_NER_ELECTRA_RTX3090_24GB/Dataset/test_data.txt` | Excluded; downloaded by notebook from the pinned dataset revision |
| `PAP_NER_ELECTRA_RTX3090_24GB/Dataset/train_data.txt` | Excluded; downloaded by notebook from the pinned dataset revision |
