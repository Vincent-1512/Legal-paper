@echo off
REM ==============================================================================
REM Script tự động chạy Giao thức Linear Evaluation Protocol trên Windows
REM Chuẩn Google SupContrast 100% (KHÔNG CÓ ALPHA)
REM ==============================================================================

set DATA_DIR=..\Dataset
set MODEL_NAME=trungbb8/distilphobert
set OUTPUT_DIR=outputs

echo ======================================================================
echo 🚀 BAT DAU CHAY LINEAR EVALUATION PROTOCOL CHO PAP_NER (GOOGLE SUPCON 100%)
echo ======================================================================

echo >>> [BUOC 1/2] Huan luyen Giai doan 1: Pure SupCon...
python main.py stage1_supcon ^
    --data_dir %DATA_DIR% ^
    --model_name_or_path %MODEL_NAME% ^
    --output_dir %OUTPUT_DIR%/stage1_supcon ^
    --batch_size 64 ^
    --learning_rate 2e-5 ^
    --head_learning_rate 5e-4 ^
    --supcon_temp 0.07 ^
    --epochs 15 ^
    --early_stop 5 ^
    --fp16

if %errorlevel% neq 0 (
    echo [ERROR] Stage 1 failed!
    exit /b %errorlevel%
)

echo >>> [BUOC 2/2] Huan luyen Giai doan 2: Linear Probe (Freeze Backbone, NO CRF)...
python main.py stage2_linear ^
    --data_dir %DATA_DIR% ^
    --model_name_or_path %MODEL_NAME% ^
    --pretrained_encoder_path %OUTPUT_DIR%/stage1_supcon/best_encoder.pt ^
    --output_dir %OUTPUT_DIR%/stage2_linear ^
    --batch_size 64 ^
    --classifier_learning_rate 1e-3 ^
    --epochs 20 ^
    --early_stop 5 ^
    --freeze_backbone ^
    --fp16

echo ======================================================================
echo [DONE] Hoan thanh Linear Evaluation Protocol!
echo Ket qua luu tai: %OUTPUT_DIR%/stage2_linear/
echo ======================================================================
