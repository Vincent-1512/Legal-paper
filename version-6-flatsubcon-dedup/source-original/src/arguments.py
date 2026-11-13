import argparse


def get_stage1_argument():
    parser = argparse.ArgumentParser(description="Stage 1: Pure Supervised Contrastive Pre-training (Google SupContrast 100%, NO ALPHA)")
    parser.add_argument("--data_dir", default="Dataset", type=str,
                        help="The input data dir. Should contain train_data.txt, dev_data.txt, test_data.txt")
    parser.add_argument("--output_dir", default="outputs/stage1_supcon", type=str,
                        help="Where to save Stage 1 pre-trained encoder checkpoint")
    parser.add_argument("--model_name_or_path", default="trungbb8/distilphobert", type=str,
                        help="Pre-trained PhoBERT model: trungbb8/distilphobert, vinai/phobert-base, vinai/phobert-large")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for SupCon training")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for validation")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="Learning rate for PhoBERT backbone in Stage 1")
    parser.add_argument("--head_learning_rate", default=5e-4, type=float,
                        help="Learning rate for Projection Head in Stage 1")
    parser.add_argument("--supcon_temp", default=0.07, type=float,
                        help="Temperature tau for SupCon loss (default: 0.07)")
    parser.add_argument("--projection_dim", default=128, type=int,
                        help="Dimension of contrastive hypersphere representation")
    parser.add_argument("--epochs", default=15, type=int,
                        help="Number of epochs for Stage 1")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Warmup proportion for linear schedule")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm")
    parser.add_argument("--early_stop", default=5, type=int,
                        help="Early stopping patience based on Validation SupCon loss")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed precision training (AMP)")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed")
    parser.add_argument("--num_workers", default=2, type=int,
                        help="DataLoader worker subprocesses")
    parser.add_argument("--overwrite_data", action="store_true", default=False,
                        help="Overwrite cached preprocessed datasets")
    return parser.parse_args()


def get_stage2_argument():
    parser = argparse.ArgumentParser(description="Stage 2: Linear Probing Evaluation Protocol (Freeze Backbone, Train Linear Classifier, NO CRF)")
    parser.add_argument("--data_dir", default="Dataset", type=str,
                        help="The input data dir.")
    parser.add_argument("--output_dir", default="outputs/stage2_linear", type=str,
                        help="Where to save Stage 2 Linear Probe model checkpoint")
    parser.add_argument("--model_name_or_path", default="trungbb8/distilphobert", type=str,
                        help="Pre-trained PhoBERT model")
    parser.add_argument("--pretrained_encoder_path", default="outputs/stage1_supcon/best_encoder.pt", type=str,
                        help="Path to pre-trained encoder from Stage 1 (SupCon)")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for Linear Probe training")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for eval")
    parser.add_argument("--classifier_learning_rate", default=1e-3, type=float,
                        help="Learning rate for Linear Classifier (default: 1e-3)")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of epochs for Stage 2")
    parser.add_argument("--early_stop", default=5, type=int,
                        help="Early stopping patience based on Validation Strict F1")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Strict Linear Evaluation Protocol: Freeze backbone parameters (requires_grad=False)")
    parser.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false",
                        help="Optional full fine-tuning mode")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed precision (AMP)")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed")
    parser.add_argument("--num_workers", default=2, type=int,
                        help="DataLoader worker subprocesses")
    parser.add_argument("--overwrite_data", action="store_true", default=False)
    return parser.parse_args()


def get_eval_argument():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 Linear Probe on Test Set")
    parser.add_argument("--data_dir", default="Dataset", type=str)
    parser.add_argument("--model_path", default="outputs/stage2_linear/best_linear_probe.pt", type=str,
                        help="Path to trained Linear Probe checkpoint")
    parser.add_argument("--model_name_or_path", default="trungbb8/distilphobert", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--output_dir", default="outputs/stage2_linear", type=str)
    return parser.parse_args()
