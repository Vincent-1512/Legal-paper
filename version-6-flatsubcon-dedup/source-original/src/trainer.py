import os
import sys
import time
import math
import datetime
import contextlib
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from .constant import LOGGER, LABEL2ID_PAP_NER, LABEL_MAPPING
from .dataset import build_dataset
from .models import PhoBertSupCon, PhoBertLinearProbe
from .conlleval import evaluate
from .helper import set_random_seed, safe_torch_load, get_total_model_parameters, plot_confusion_matrix


def _get_amp_context(use_amp: bool, device: torch.device):
    """Bọc autocast tương thích toàn diện cho PyTorch 1.x đến 2.6+ trên GPU Colab/Server."""
    if use_amp and str(device).startswith('cuda'):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast('cuda')
        elif hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def _get_grad_scaler(use_amp: bool, device: torch.device):
    """Bọc GradScaler tương thích toàn diện cho PyTorch 1.x đến 2.6+."""
    if use_amp and str(device).startswith('cuda'):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            return torch.amp.GradScaler('cuda')
        elif hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
            return torch.cuda.amp.GradScaler()
    return None


def train_stage1_supcon(args):
    """
    Huấn luyện Stage 1: Tối ưu không gian biểu diễn bằng Supervised Contrastive Loss phẳng chuẩn Google 100%.
    Hoàn toàn KHÔNG có hệ số alpha.
    Không dùng Cross-Entropy, không dùng CRF.
    Lưu trọng số encoder tốt nhất tại outputs/stage1_supcon/best_encoder.pt.
    """
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not getattr(args, 'no_cuda', False) else 'cpu')

    LOGGER.info("=" * 65)
    LOGGER.info("🚀 BẮT ĐẦU HUẤN LUYỆN GIAI ĐOẠN 1: PURE SUPERVISED CONTRASTIVE (Google 100%, NO ALPHA)")
    LOGGER.info(f"👉 Model Backbone:     {args.model_name_or_path}")
    LOGGER.info(f"👉 Temperature (tau):  {args.supcon_temp}")
    LOGGER.info(f"👉 Batch size:         {args.batch_size}")
    LOGGER.info(f"👉 Backbone LR:        {args.learning_rate} | Head LR: {args.head_learning_rate}")
    LOGGER.info(f"👉 Device:             {device}")
    LOGGER.info("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # 1. Nạp Dataset
    train_dataset = build_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        label2id=LABEL2ID_PAP_NER,
        header=LABEL_MAPPING['pap_ner']['header'],
        dtype='train',
        max_seq_len=args.max_seq_length,
        overwrite_data=args.overwrite_data
    )
    dev_dataset = build_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        label2id=LABEL2ID_PAP_NER,
        header=LABEL_MAPPING['pap_ner']['header'],
        dtype='dev',
        max_seq_len=args.max_seq_length,
        overwrite_data=args.overwrite_data
    )

    pin_mem = True if str(device).startswith('cuda') else False
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )
    dev_loader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )

    # 2. Khởi tạo Mô hình PhoBertSupCon nạp pretrained weights từ HuggingFace
    model = PhoBertSupCon(
        model_name_or_config=args.model_name_or_path,
        projection_dim=args.projection_dim,
        temperature=args.supcon_temp,
        o_label_id=0
    ).to(device)

    total_p, train_p = get_total_model_parameters(model)
    LOGGER.info(f"Tổng tham số: {total_p:,} | Tham số huấn luyện: {train_p:,}")

    # 3. Optimizer & Scheduler
    optimizer_grouped_parameters = [
        {"params": model.roberta.parameters(), "lr": args.learning_rate},
        {"params": model.projection_head.parameters(), "lr": args.head_learning_rate}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = args.fp16 and str(device).startswith('cuda')
    scaler = _get_grad_scaler(use_amp, device)

    # 4. Huấn luyện qua các Epochs
    best_dev_loss = float('inf')
    early_stop_count = 0
    best_model_path = os.path.join(args.output_dir, 'best_encoder.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            with _get_amp_context(use_amp, device):
                outputs = model(**batch)
                loss = outputs.loss

            if loss.item() > 0:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

            scheduler.step()
            loss_val = loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                train_loss += loss_val
                train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        # Validation SupCon loss
        model.eval()
        dev_loss = 0.0
        dev_steps = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with _get_amp_context(use_amp, device):
                    outputs = model(**batch)
                loss_val = outputs.loss.item()
                if not math.isnan(loss_val) and not math.isinf(loss_val):
                    dev_loss += loss_val
                    dev_steps += 1
        avg_dev_loss = dev_loss / max(dev_steps, 1)
        epoch_dur = datetime.timedelta(seconds=int(time.time() - epoch_start))

        LOGGER.info(f"[STAGE 1 - Epoch {epoch:02d}/{args.epochs:02d}] "
                    f"Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | Time: {epoch_dur}")

        # Kiểm tra cải thiện & Lưu checkpoint encoder
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            early_stop_count = 0
            checkpoint_data = {
                'epoch': epoch,
                'encoder_state_dict': model.roberta.state_dict(),
                'projection_head_state_dict': model.projection_head.state_dict(),
                'dev_loss': avg_dev_loss,
                'args': vars(args)
            }
            torch.save(checkpoint_data, best_model_path)
            LOGGER.info(f"  ⭐ Đã lưu best encoder mới tại: {best_model_path} (Dev Loss: {best_dev_loss:.4f})")
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                LOGGER.info(f"  🛑 Early stopping kích hoạt tại epoch {epoch} sau {args.early_stop} epoch không cải thiện.")
                break

    LOGGER.info(f"✅ Hoàn thành Stage 1! Checkpoint lưu tại: {best_model_path}")
    return best_model_path


def train_stage2_linear(args):
    """
    Huấn luyện Stage 2: Linear Probing Evaluation Protocol (chuẩn Google SupContrast).
    - Khóa chặt toàn bộ tham số của PhoBERT Backbone (with torch.no_grad() & requires_grad=False).
    - Chỉ huấn luyện tầng Linear Classifier (nn.Linear(768, 11)) với CrossEntropyLoss.
    - KHÔNG DÙNG CRF.
    - Đánh giá và dừng sớm dựa trên Validation Strict Entity-level F1 (CoNLL-2003 metric).
    """
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not getattr(args, 'no_cuda', False) else 'cpu')

    LOGGER.info("=" * 65)
    LOGGER.info("🚀 BẮT ĐẦU HUẤN LUYỆN GIAI ĐOẠN 2: LINEAR PROBING EVALUATION PROTOCOL")
    LOGGER.info(f"👉 Model Backbone:         {args.model_name_or_path}")
    LOGGER.info(f"👉 Pre-trained Encoder:    {args.pretrained_encoder_path}")
    LOGGER.info(f"👉 Freeze Backbone:        {args.freeze_backbone} (Strict Linear Probe)")
    LOGGER.info(f"👉 Classifier LR:          {args.classifier_learning_rate}")
    LOGGER.info(f"👉 Batch size:             {args.batch_size}")
    LOGGER.info(f"👉 Device:                 {device}")
    LOGGER.info("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # 1. Nạp Dataset
    train_dataset = build_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        label2id=LABEL2ID_PAP_NER,
        header=LABEL_MAPPING['pap_ner']['header'],
        dtype='train',
        max_seq_len=args.max_seq_length,
        overwrite_data=args.overwrite_data
    )
    dev_dataset = build_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        label2id=LABEL2ID_PAP_NER,
        header=LABEL_MAPPING['pap_ner']['header'],
        dtype='dev',
        max_seq_len=args.max_seq_length,
        overwrite_data=args.overwrite_data
    )
    test_dataset = build_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        label2id=LABEL2ID_PAP_NER,
        header=LABEL_MAPPING['pap_ner']['header'],
        dtype='test',
        max_seq_len=args.max_seq_length,
        overwrite_data=args.overwrite_data
    )

    pin_mem = True if str(device).startswith('cuda') else False
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )
    dev_loader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )

    # 2. Khởi tạo PhoBertLinearProbe
    model = PhoBertLinearProbe(
        model_name_or_config=args.model_name_or_path,
        num_labels=len(LABEL2ID_PAP_NER),
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # Nạp trọng số pretrained encoder từ Stage 1 nếu có
    if args.pretrained_encoder_path and os.path.isfile(args.pretrained_encoder_path):
        LOGGER.info(f"📥 Nạp trọng số encoder từ Stage 1: {args.pretrained_encoder_path}")
        checkpoint = safe_torch_load(args.pretrained_encoder_path, map_location=device)
        if 'encoder_state_dict' in checkpoint:
            model.roberta.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'model' in checkpoint:
            model.roberta.load_state_dict(checkpoint['model'])
        else:
            model.roberta.load_state_dict(checkpoint)
        LOGGER.info("✅ Nạp trọng số encoder thành công!")
    else:
        LOGGER.warning(f"⚠️ Không tìm thấy pretrained encoder tại: {args.pretrained_encoder_path}. Khởi tạo từ pre-trained backbone mặc định!")

    total_p, train_p = get_total_model_parameters(model)
    LOGGER.info(f"Tổng tham số: {total_p:,} | Tham số huấn luyện (Chỉ Linear Head): {train_p:,}")

    # 3. Optimizer: Chỉ tối ưu classifier parameters khi freeze_backbone=True
    if args.freeze_backbone:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.classifier_learning_rate)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.roberta.parameters(), "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": args.classifier_learning_rate}
        ])

    use_amp = args.fp16 and str(device).startswith('cuda')
    scaler = _get_grad_scaler(use_amp, device)

    # 4. Huấn luyện qua các Epochs
    best_dev_f1 = 0.0
    early_stop_count = 0
    best_model_path = os.path.join(args.output_dir, 'best_linear_probe.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.roberta.eval()  # Đảm bảo backbone luôn ở trạng thái eval
        epoch_start = time.time()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            with _get_amp_context(use_amp, device):
                outputs = model(**batch)
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_val = loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                train_loss += loss_val
                train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        # Đánh giá trên tập Dev bằng CoNLL-2003 Strict Entity-level F1
        dev_prec, dev_rec, dev_f1 = evaluate_linear_probe(
            model=model,
            iterator=dev_loader,
            device=device,
            is_test=False
        )

        epoch_dur = datetime.timedelta(seconds=int(time.time() - epoch_start))
        LOGGER.info(f"[STAGE 2 - Epoch {epoch:02d}/{args.epochs:02d}] "
                    f"Train Loss: {avg_train_loss:.4f} | Dev Strict P: {dev_prec:.4f} | R: {dev_rec:.4f} | F1: {dev_f1:.4f} | Time: {epoch_dur}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            early_stop_count = 0
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classes': LABEL2ID_PAP_NER,
                'best_dev_f1': best_dev_f1,
                'args': vars(args)
            }
            torch.save(checkpoint_data, best_model_path)
            LOGGER.info(f"  ⭐ Đã lưu best Linear Probe checkpoint tại: {best_model_path} (Dev Strict F1: {best_dev_f1:.4f})")
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                LOGGER.info(f"  🛑 Early stopping kích hoạt tại epoch {epoch} sau {args.early_stop} epoch không cải thiện.")
                break

    LOGGER.info("=" * 65)
    LOGGER.info(f"🏁 ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST VỚI CHECKPOINT TỐT NHẤT")
    LOGGER.info("=" * 65)
    best_ckpt = safe_torch_load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_prec, test_rec, test_f1 = evaluate_linear_probe(
        model=model,
        iterator=test_loader,
        device=device,
        is_test=True,
        output_dir=args.output_dir
    )
    LOGGER.info(f"🎯 KẾT QUẢ TEST CUỐI CÙNG:")
    LOGGER.info(f"   Strict Precision: {test_prec:.4f}")
    LOGGER.info(f"   Strict Recall:    {test_rec:.4f}")
    LOGGER.info(f"   Strict Micro-F1:  {test_f1:.4f}")
    LOGGER.info("=" * 65)
    return test_prec, test_rec, test_f1


def evaluate_linear_probe(model, iterator, device='cuda', is_test: bool = False,
                          output_dir: str = './outputs/stage2_linear') -> Tuple[float, float, float]:
    """
    Đánh giá mô hình Linear Probe theo chuẩn CoNLL-2003 Entity-level Strict Metric.
    """
    model.eval()
    eval_golds = []
    eval_preds = []

    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)

            active_mask = (batch['label_masks'] != 0)
            labels = batch['labels']
            batch_tags = outputs.tags

            for b in range(len(batch_tags)):
                length = active_mask[b].sum().item()
                gold_seq = labels[b, :length].cpu().tolist()
                pred_seq = batch_tags[b]
                eval_golds.extend(gold_seq)
                eval_preds.extend(pred_seq)

    id2label = LABEL_MAPPING['pap_ner']['id2label']
    gold_str = [id2label[tag] for tag in eval_golds]
    pred_str = [id2label[tag] for tag in eval_preds]

    prec, rec, f1 = evaluate(gold_str, pred_str, verbose=is_test)

    if is_test:
        report = classification_report(
            eval_golds, eval_preds,
            target_names=LABEL2ID_PAP_NER,
            digits=4,
            zero_division=0
        )
        LOGGER.info("\n" + report)
        plot_confusion_matrix(
            golds=eval_golds,
            preds=eval_preds,
            classes=LABEL2ID_PAP_NER,
            output_dir=output_dir,
            title='Normalized Confusion Matrix (Linear Probe Stage 2)',
            normalize=True
        )

    return prec, rec, f1
