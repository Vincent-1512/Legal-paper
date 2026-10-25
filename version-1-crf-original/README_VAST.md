# PAP_NER ELECTRA-CRF — RTX 3090 24 GB trên Vast.ai

## Upload và chạy

Upload PAP_NER_ELECTRA_RTX3090_24GB.zip vào Jupyter, rồi giải nén:

```bash
python -m zipfile -e PAP_NER_ELECTRA_RTX3090_24GB.zip .
```

Mở PAP_NER_ELECTRA_RTX3090_24GB/PAP_NER_ELECTRA_CRF_20epoch_RTX3090_24GB_evalplus_v2.ipynb.
Để dùng dataset local đóng gói, thay CUSTOM_DATA_DIR=None trong cell cấu hình bằng:

```python
CUSTOM_DATA_DIR = Path.cwd() / "Dataset"
```

Giữ GPU_PROFILE="rtx3090_24gb", RESUME=False khi bắt đầu run mới. Chạy từng cell, qua GPU check và smoke test trước full train. Cần Internet để cài dependency và tải pretrained ELECTRA.

## Cấu hình

RTX3090 24GB, train/eval batch32, accumulation1, effectivebatch32, gradient checkpointing tắt, FP16. Model NlpHUST/electra-base-vn + CRF; tối đa20epochs, earlystop3, seed42, maxlength256, encoder LR2e-5, head LR1e-3. Các tham số học khác và phần đánh giá giữ nguyên.

## Kết quả đã tải về

Thư mục papner_electra_crf_official_20ep_seed42_rtx3090_24gb và ZIP cùng tên -results.zip đã ghi đúng cấu hình RTX3090, nên giữ nguyên tên. best_model.pt, tokenizer, metrics và config gốc không bị sửa. Không đổi tên các file checkpoint chuẩn vì runner sử dụng chúng.

Cấu hình thực tế tối đa20epochs; lượt đã chạy dừng ở epoch6, dev tốt nhất epoch3. test Strict MicroF1=97.7017%, Strict MacroF1=97.8456%. Đây là kết quả lượt đã tải, không phải một lượt train mới.

ZIP upload này chứa notebook và dataset để chạy lại; ZIP -results chứa model/báo cáo đã train. last_checkpoint.pt không nằm trong ZIP -results hiện tại. Bản notebook trước đổi tên được lưu tại backups trên máy local.
