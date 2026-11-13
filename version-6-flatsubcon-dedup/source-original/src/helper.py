import os
import random
import logging
import argparse
from typing import List, Optional

import torch
import numpy as np

# Safe globals for PyTorch 2.6+
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass


def safe_torch_load(f, map_location=None, **kwargs):
    """Safely loads torch files supporting PyTorch 2.6+ where weights_only defaults to True."""
    try:
        return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(f, map_location=map_location, **kwargs)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(log_file: Optional[str] = None, log_file_level=logging.INFO):
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("baseline_supcon")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    if log_file:
        try:
            log_dir = './logs'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(log_file_level)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not initialize file logging: {e}")
    return logger


def get_total_model_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def plot_confusion_matrix(golds: List[int], preds: List[int], classes: List[str],
                          output_dir: str = './outputs', title: str = 'Confusion Matrix',
                          normalize: bool = True):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(golds, preds, labels=list(range(len(classes))))
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        pass

