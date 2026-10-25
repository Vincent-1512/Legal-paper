# Lần 4 — ELECTRA + CRF + Hierarchical SupCon

Notebook cần mở: PAP_NER_ELECTRA_CRF_HierarchicalSupCon_dedup_20epoch_RTX3090_24GB.ipynb.

## Chạy trên Vast

1. Chọn instance Linux RTX 3090 24 GB với Jupyter và Python 3.10–3.12 có venv/ensurepip.
2. Upload ZIP PAP_NER_ELECTRA_HSUPCON_VAST_RTX3090_24GB.zip và giải nén, rồi mở notebook mới ở trên.
3. Chạy từ đầu đến hết mục 9. Notebook tự cài thư viện, tải model và dataset từ Hugging Face; cần Internet. Không cần Google Drive hay VPhoBertTagger.
4. Kiểm tra smoke test ở mục 9 thành công, loss hữu hạn; sau đó chạy mục 10 để train và mục 11 để theo dõi.
5. Sau khi train xong, chạy các mục 12–16 để vẽ đồ thị, đánh giá test và tải ZIP kết quả.

## Cấu hình

- Encoder: NlpHUST/electra-base-vn, giữ nguyên lần 3.
- Dataset dedup 70/15/15: levandai1704/PAP_NER_dedup_70_15_15; train/dev/test 24.089/5.162/5.162 câu, revision e4af303d22f878e5827c94861f50f747a6b3ed45.
- Batch 32, gradient accumulation 1, FP16, max length 256, seed 42, tối đa 20 epochs; early stop 3 theo dev Strict Micro F1.
- Projection Head trước dropout: hidden_size → 256 → 128; chuẩn hóa L2 trong SupCon.
- Hierarchical positives: cùng BIO 1.0, cùng entity khác B/I 0.3, khác entity 0.0. Loại O/padding/self; anchor không có positive không tham gia loss.
- Loss train: CRF mean + 0.3 × SupCon mean; temperature 0.07. Hai loss tính FP32; SupCon không chạy khi eval/test/inference.
- LR encoder 2e-5; classifier/CRF/projection 1e-3. Log train_crf_loss_per_sentence, train_supcon_loss và train_loss_per_sentence.

## Đối chứng và checkpoint

USE_SUPCON=True là mặc định. Đổi False rồi chạy từ đầu sẽ tạo run CRF mean riêng để đối chứng. Lần 3 dùng CRF sum nên kết quả so sánh trực tiếp có thêm khác biệt reduction. Batch tương phản thực tế là 32; accumulation không ghép tập cặp SupCon giữa các microbatch.

Run và runner có tên riêng. RESUME=False để train mới. Chỉ resume checkpoint cùng cấu hình; không dùng checkpoint lần 3 để resume. ZIP kết quả chứa best_model.pt, tokenizer, config, lịch sử, báo cáo, dự đoán và các đồ thị; không chứa last_checkpoint.pt.

## Kiểm tra

Hai script trong tests/ kiểm tra trên CPU: công thức loss/gradient/optimizer/checkpoint và toàn bộ runner với mô hình ELECTRA nhỏ, dữ liệu giả, không tải model. Việc này không thay thế smoke test CUDA và kiểm tra VRAM trên instance Vast.
