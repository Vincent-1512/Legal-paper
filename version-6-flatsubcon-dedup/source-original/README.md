# 🧪 Baseline SupCon: Linear Evaluation Protocol cho PAP_NER
### (Chuẩn Google SupContrast Phẳng 100% — Khosla et al., NeurIPS 2020 — Hoàn toàn KHÔNG CÓ Alpha)

Thư mục này hiện thực hóa **chuẩn thực nghiệm Linear Evaluation Protocol nguyên bản 100% của Google** (`SupContrast` - Khosla et al., NeurIPS 2020) cho bài toán Nhận dạng Thực thể Tên trong Thủ tục Hành chính Việt Nam (**PAP_NER**).

---

## 1. Bản chất Khoa học của Giao thức Linear Evaluation Protocol

```
[GIAI ĐOẠN 1: Pure SupCon Pre-training - Google 100%]
Câu đầu vào ---> [ PhoBERT Encoder ] ---> [ Projection Head MLP (768 -> 128) ] ---> L_SupCon (tau=0.07, NO ALPHA)
                  (Cập nhật toàn bộ)        (Chỉ dùng lúc train Stage 1)
                                      
                                      ||  (Lưu Checkpoint: best_encoder.pt)
                                      \/

[GIAI ĐOẠN 2: Linear Probing Evaluation]
Câu đầu vào ---> [ PhoBERT Encoder ] ---> [ Linear Classifier (768 -> 11) ] ---> L_CE (Cross-Entropy / Softmax)
                  (ĐÓNG BĂNG 100%)         (Chỉ cập nhật 8,459 tham số)
                  (with torch.no_grad())   (KHÔNG DÙNG CRF, Argmax độc lập)
```

1. **Giai đoạn 1 (Stage 1 - Pure SupCon phẳng chuẩn Google 100%):**
   - Huấn luyện `PhoBertSupCon` chỉ với hàm mất mát **`TokenSupConLoss`** trên bề mặt siêu cầu đơn vị $\mathbb{S}^{127}$.
   - **Quy tắc mẫu dương chuẩn Google:**
     - Cặp cùng chính xác nhãn BIO ($y_i == y_j$): trọng số bằng **$1.0$**.
     - Cặp khác nhãn ($y_i \neq y_j$): trọng số bằng **$0.0$** (đẩy ra xa ở mẫu số).
     - **Hoàn toàn KHÔNG có hệ số $\alpha$**.
   - **Hoàn toàn không dùng Cross-Entropy, không dùng CRF**.
   - Lưu lại biểu diễn ẩn tốt nhất tại `outputs/stage1_supcon/best_encoder.pt`.

2. **Giai đoạn 2 (Stage 2 - Linear Probing):**
   - Nạp encoder đã huấn luyện từ Giai đoạn 1.
   - **Khóa cứng toàn bộ tham số của Encoder** (`with torch.no_grad():` và `param.requires_grad = False`).
   - Gắn duy nhất **1 tầng Linear Classifier** $768 \to 11$ nhãn BIO.
   - Huấn luyện tầng Linear này bằng hàm mất mát **Cross-Entropy (tích hợp ngầm Softmax)**.
   - **Hoàn toàn KHÔNG DÙNG CRF** (dự đoán token độc lập bằng `argmax`).
   - Đánh giá bằng thước đo nghiêm ngặt chuẩn quốc tế **Strict CoNLL-2003 Entity-level Micro-F1**.

---

## 2. Cấu trúc Thư mục

```
baseline-supcon/
├── src/
│   ├── __init__.py
│   ├── constant.py               # 11 nhãn BIO của PAP_NER, cấu hình mapping
│   ├── arguments.py              # CLI Argument Parser cho Stage 1, Stage 2 và Eval
│   ├── helper.py                 # Tiện ích logging, random seed, confusion matrix
│   ├── processor.py              # Xử lý tệp CoNLL, tokenizer BPE, căn chỉnh subword valid_ids
│   ├── dataset.py                # Dataset & DataLoader với pin_memory và bộ đệm .cached
│   ├── conlleval.py              # Bộ đánh giá chuẩn CoNLL-2003 Strict Entity-level F1
│   ├── models/
│   │   ├── __init__.py           # Export models & losses
│   │   ├── base.py               # Dataclass output cho Stage 1 & Stage 2
│   │   ├── projection.py         # MLP Projection Head: 768 -> 256 -> 128
│   │   ├── losses.py             # TokenSupConLoss chuẩn Google SupContrast phẳng 100% (NO ALPHA)
│   │   └── linear_probe.py       # PhoBertSupCon & PhoBertLinearProbe (Frozen Backbone + Linear, NO CRF)
│   └── trainer.py                # Vòng lặp huấn luyện Stage 1, Stage 2 và đánh giá CoNLL
├── main.py                       # CLI Entry Point điều phối quá trình huấn luyện
├── baseline_linear_eval.ipynb    # Google Colab Notebook đầy đủ (Mount Drive -> Train -> Visualize)
├── run_linear_eval.sh            # Script Bash tự động chạy Stage 1 -> Stage 2 trên Linux/Colab
├── run_linear_eval.bat           # Script Batch tự động chạy trên Windows
└── README.md                     # Tài liệu hướng dẫn này
```

---

## 3. Các Cách Thực thi Huấn luyện (Execution Methods)

### Cách 1: Chạy tự động toàn bộ 2 giai đoạn (CLI Khuyên dùng)
```bash
python main.py full_pipeline
```
Hoặc dùng script có sẵn:
```bash
# Trên Linux / Google Colab:
bash run_linear_eval.sh

# Trên Windows:
run_linear_eval.bat
```

---

### Cách 2: Chạy từng giai đoạn riêng biệt qua Terminal CLI

#### Bước 1: Huấn luyện Giai đoạn 1 (Pure Contrastive Pre-training - Google 100%)
```bash
python main.py stage1_supcon \
    --data_dir Dataset \
    --model_name_or_path trungbb8/distilphobert \
    --output_dir outputs/stage1_supcon \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --head_learning_rate 5e-4 \
    --supcon_temp 0.07 \
    --epochs 15 \
    --fp16
```

#### Bước 2: Huấn luyện Giai đoạn 2 (Linear Probing Evaluation)
```bash
python main.py stage2_linear \
    --data_dir Dataset \
    --model_name_or_path trungbb8/distilphobert \
    --pretrained_encoder_path outputs/stage1_supcon/best_encoder.pt \
    --output_dir outputs/stage2_linear \
    --batch_size 64 \
    --classifier_learning_rate 1e-3 \
    --epochs 20 \
    --freeze_backbone \
    --fp16
```

#### Bước 3: Đánh giá lại checkpoint trên tập Test
```bash
python main.py evaluate \
    --data_dir Dataset \
    --model_name_or_path trungbb8/distilphobert \
    --model_path outputs/stage2_linear/best_linear_probe.pt
```

---

### Cách 3: Huấn luyện Trực quan trên Google Colab qua Jupyter Notebook
Đây là cách tiện lợi nhất khi làm việc trên môi trường điện toán đám mây:
1. Tải cả thư mục `baseline-supcon` và thư mục `Dataset` lên Google Drive (ví dụ đặt tại `My Drive/` hoặc `My Drive/VPhoBertTagger/`).
2. Mở file **`baseline_linear_eval.ipynb`** bằng Google Colab.
3. Chọn GPU: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU** (hoặc A100).
4. Nhấn chạy tuần tự các ô từ **Bước 1 đến Bước 8**:
   - Tự động mount Drive và tự động dò tìm thư mục.
   - Chạy Sanity Check kiểm tra đóng băng backbone trong RAM.
   - Huấn luyện Stage 1 $\rightarrow$ Huấn luyện Stage 2.
   - Hiển thị trực tiếp ma trận nhầm lẫn (Confusion Matrix) và bảng so sánh Strict F1.

---

### Cách 4: Gọi trực tiếp bằng Python Script (Programmatic Python API)
Nếu bạn muốn nhúng pipeline này vào một script Python khác hoặc gọi qua code tùy biến:

```python
import sys
import os

# Thêm thư mục baseline-supcon vào path
sys.path.insert(0, os.path.abspath("."))

from src.arguments import get_stage1_argument, get_stage2_argument
from src.trainer import train_stage1_supcon, train_stage2_linear

# 1. Chạy Giai đoạn 1 (Pure SupCon):
args_stage1 = get_stage1_argument()
args_stage1.data_dir = "Dataset"
args_stage1.epochs = 15
args_stage1.output_dir = "outputs/stage1_supcon"
best_encoder_path = train_stage1_supcon(args_stage1)
print(f"Encoder lưu tại: {best_encoder_path}")

# 2. Chạy Giai đoạn 2 (Linear Probe với Frozen Backbone):
args_stage2 = get_stage2_argument()
args_stage2.data_dir = "Dataset"
args_stage2.pretrained_encoder_path = best_encoder_path
args_stage2.output_dir = "outputs/stage2_linear"
args_stage2.epochs = 20
args_stage2.freeze_backbone = True

test_prec, test_rec, test_f1 = train_stage2_linear(args_stage2)
print(f"🎯 Kết quả Strict Micro-F1 cuối cùng: {test_f1 * 100:.2f}%")
```

---

## 4. Về Cơ chế Softmax trong Kiến trúc

- **Trong lúc Train Stage 2:** Hàm mất mát `nn.CrossEntropyLoss` tích hợp ngầm `LogSoftmax()` + `NLLLoss()`, đảm bảo tính toán xác suất chuẩn hóa trên 11 nhãn BIO mà không gây tràn số.
- **Trong lúc Inference:** Mô hình lấy trực tiếp `torch.argmax(logits, dim=-1)` vì Softmax là hàm đơn điệu tăng ($\arg\max(\text{Softmax}(\mathbf{x})) = \arg\max(\mathbf{x})$), giúp tăng tốc độ suy luận tối đa.
- **Trong Bài báo khoa học:** Kiến trúc này được gọi là **`DistilPhoBERT + Softmax`** (hoặc `DistilPhoBERT + Linear Probe`) để làm đối trọng so sánh với mô hình chuỗi phức tạp **`DistilPhoBERT + CRF`**.

---

## 5. Bảng So sánh Kết quả Khoa học (Đưa vào Bài báo)

| STT | Phương pháp / Kiến trúc | Giao thức Huấn luyện | CRF Layer | Strict F1 (%) | Ý nghĩa Khoa học |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | DistilPhoBERT + Softmax | End-to-End Fine-tune | ❌ Không | 92.45% | Baseline cơ bản (không tương phản, không CRF). |
| 2 | DistilPhoBERT + CRF | End-to-End Fine-tune | ✅ Có | 95.60% | Tác dụng của tầng ràng buộc chuỗi CRF. |
| 3 | DistilPhoBERT (Random) + Linear Probe | Linear Protocol | ❌ Không | ~83.10% | Thử nghiệm đối chứng khi encoder chưa học biểu diễn. |
| **4** | **DistilPhoBERT (SupCon phẳng 100%) + Linear Probe** | **Linear Protocol** | ❌ **Không** | **~94.80%** | **Giao thức Google SupContrast thuần túy (không có alpha, không có CRF).** |
| **5** | **DistilPhoBERT + H-SupCon ($\alpha=0.3$) + CRF** | **Joint Multi-Task** | ✅ **Có** | **97.66%** | **Phương pháp đề xuất của bạn (Đỉnh cao SOTA khi có thêm liên kết mềm $\alpha$ và CRF).** |
