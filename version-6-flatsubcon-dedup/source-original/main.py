import os
import sys

# Đảm bảo đường dẫn thư mục hiện tại được ưu tiên hàng đầu trong sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from src.constant import LOGGER
from src.arguments import get_stage1_argument, get_stage2_argument, get_eval_argument
from src.trainer import train_stage1_supcon, train_stage2_linear, evaluate_linear_probe
from src.models import PhoBertLinearProbe
from src.dataset import build_dataset
from src.helper import safe_torch_load
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, SequentialSampler
import torch


def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("📌 GIAO THỨC LINEAR EVALUATION CHO PAP_NER (PhoBERT + SupCon)")
        print("Cách sử dụng:")
        print("  1. Huấn luyện Giai đoạn 1 (Pure Contrastive):")
        print("     python main.py stage1_supcon --data_dir Dataset --epochs 15")
        print("  2. Huấn luyện Giai đoạn 2 (Linear Probing Evaluation, Freeze Backbone, NO CRF):")
        print("     python main.py stage2_linear --data_dir Dataset --epochs 20")
        print("  3. Đánh giá Checkpoint Giai đoạn 2 trên tập Test:")
        print("     python main.py evaluate --model_path outputs/stage2_linear/best_linear_probe.pt")
        print("  4. Chạy toàn bộ cả 2 Giai đoạn tự động:")
        print("     python main.py full_pipeline")
        print("=" * 70)
        sys.exit(0)

    mode = sys.argv[1].lower()
    # Loại bỏ tên chế độ khỏi sys.argv để ArgumentParser parse các cờ còn lại
    sys.argv.pop(1)

    if mode == 'stage1_supcon':
        args = get_stage1_argument()
        train_stage1_supcon(args)

    elif mode == 'stage2_linear':
        args = get_stage2_argument()
        train_stage2_linear(args)

    elif mode == 'full_pipeline':
        LOGGER.info("🚀 BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE (STAGE 1 -> STAGE 2)...")
        # 1. Chạy Stage 1
        args_stage1 = get_stage1_argument()
        best_encoder_path = train_stage1_supcon(args_stage1)

        # 2. Chạy Stage 2 sử dụng encoder vừa huấn luyện
        args_stage2 = get_stage2_argument()
        args_stage2.pretrained_encoder_path = best_encoder_path
        train_stage2_linear(args_stage2)

    elif mode == 'evaluate':
        args = get_eval_argument()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOGGER.info(f"Nạp checkpoint đánh giá: {args.model_path}")
        checkpoint = safe_torch_load(args.model_path, map_location=device)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

        model = PhoBertLinearProbe(config, num_labels=11, freeze_backbone=True).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        from src.constant import LABEL2ID_PAP_NER, LABEL_MAPPING
        test_dataset = build_dataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            label2id=LABEL2ID_PAP_NER,
            header=LABEL_MAPPING['pap_ner']['header'],
            dtype='test',
            max_seq_len=args.max_seq_length
        )
        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        evaluate_linear_probe(model, test_loader, device=device, is_test=True, output_dir=args.output_dir)

    else:
        LOGGER.error(f"❌ Chế độ '{mode}' không hợp lệ! Vui lòng chọn: stage1_supcon, stage2_linear, full_pipeline, evaluate.")


if __name__ == '__main__':
    main()

