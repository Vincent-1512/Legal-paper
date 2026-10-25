# Chạy PAP_NER ELECTRA + Softmax trên Vast

## File trong gói
- PAP_NER_MultiModel_Softmax_evalplus_v1.ipynb: bản sao nguyên byte của notebook hiện tại.
- README_VAST.md: hướng dẫn chạy.

Notebook tự tạo runner, môi trường Python, tải model/dataset và tạo các thư mục cần thiết. Không cần upload checkpoint, kết quả CRF, thư mục backup hay môi trường Python từ máy Mac.

## Môi trường theo cấu hình notebook
- Linux có NVIDIA driver, nvidia-smi và GPU RTX 3090 24 GB.
- Jupyter với Python 3.10–3.12 có venv/ensurepip.
- Có kết nối Internet để tải các package, model và dataset từ Hugging Face.
- Notebook kiểm tra tối thiểu 12 GiB dung lượng trống; cần dành thêm dung lượng cho checkpoint và kết quả.

## Cách chạy
1. Upload file ZIP lên máy Vast qua Jupyter, rồi giải nén.
2. Mở PAP_NER_MultiModel_Softmax_evalplus_v1.ipynb trong thư mục vừa giải nén.
3. Chạy các cell lần lượt từ trên xuống. Cell cấu hình phải in:
   Model choice: electra (FPTAI/velectra-base-discriminator-cased)
4. Chạy cell smoke test ở mục 9 và kiểm tra hoàn tất thành công trước khi chạy cell train ở mục 10.
5. Theo dõi training ở mục 11, rồi chạy các mục đánh giá và đóng gói kết quả.
6. Tải file *-results.zip được tạo ở mục 16 về máy trước khi kết thúc instance.

## Cấu hình giữ nguyên
- ELECTRA: FPTAI/velectra-base-discriminator-cased.
- Linear classification head + softmax/cross-entropy.
- 20 epochs; early stopping patience 5 có thể dừng sớm.
- Batch 32, gradient accumulation 1, FP16, seed 42.
- Dataset Leekien0108/PAP_NER, revision 7a0f6b1233667702925baa69c07c55d5d26eb6cd.
- RESUME=False: lần train mới.

Nếu /workspace tồn tại, notebook mặc định ghi dữ liệu, môi trường và run vào /workspace/ner-pap. Nếu không, nó dùng ner-pap trong thư mục làm việc.

Gói đã kiểm tra tính toàn vẹn ZIP và notebook giữ nguyên. Chưa chạy smoke test hay training trên GPU Vast.
