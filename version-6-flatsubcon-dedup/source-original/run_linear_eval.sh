#!/bin/bash
# ==============================================================================
# Script tự động chạy Giao thức Linear Evaluation Protocol (Stage 1 -> Stage 2)
# Chuẩn Google SupContrast 100% (KHÔNG CÓ ALPHA)
# ==============================================================================

set -e

DATA_DIR="../Dataset"
MODEL_NAME="trungbb8/distilphobert"
OUTPUT_DIR="outputs"

echo "======================================================================"
echo "🚀 BẮT ĐẦU CHẠY LINEAR EVALUATION PROTOCOL CHO PAP_NER (GOOGLE SUPCON 100%)"
echo "======================================================================"

# Giai đoạn 1: Pure Contrastive Pretraining (Google SupContrast)
echo ">>> [BƯỚC 1/2] Huấn luyện Giai đoạn 1: Pure SupCon..."
python main.py stage1_supcon \
    --data_dir "$DATA_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/stage1_supcon" \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --head_learning_rate 5e-4 \
    --supcon_temp 0.07 \
    --epochs 15 \
    --early_stop 5 \
    --fp16

# Giai đoạn 2: Linear Probing Evaluation
echo ">>> [BƯỚC 2/2] Huấn luyện Giai đoạn 2: Linear Probe (Freeze Backbone, NO CRF)..."
python main.py stage2_linear \
    --data_dir "$DATA_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --pretrained_encoder_path "$OUTPUT_DIR/stage1_supcon/best_encoder.pt" \
    --output_dir "$OUTPUT_DIR/stage2_linear" \
    --batch_size 64 \
    --classifier_learning_rate 1e-3 \
    --epochs 20 \
    --early_stop 5 \
    --freeze_backbone \
    --fp16

echo "======================================================================"
echo "✅ HOÀN THÀNH TOÀN BỘ GIAO THỨC LINEAR EVALUATION PROTOCOL!"
echo "Kết quả lưu tại: $OUTPUT_DIR/stage2_linear/"
echo "======================================================================"
