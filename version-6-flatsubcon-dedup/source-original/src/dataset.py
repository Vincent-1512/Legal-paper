import os
from pathlib import Path
from typing import List, Union
import torch
from torch.utils.data import Dataset

from .processor import NerFeatures, convert_word_segment_examples_features
from .helper import safe_torch_load

if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([NerFeatures])
    except Exception:
        pass


class NerDataset(Dataset):
    def __init__(self, features: List[NerFeatures]):
        self.examples = features

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: val for key, val in self.examples[index].__dict__.items()}


def build_dataset(data_dir: Union[str, os.PathLike],
                  tokenizer,
                  label2id: List[str],
                  header: List[str],
                  dtype: str = 'train',
                  max_seq_len: int = 256,
                  overwrite_data: bool = False) -> NerDataset:
    """
    Nạp và xử lý tập dữ liệu PAP_NER với cơ chế tìm kiếm đa đường dẫn
    đảm bảo chạy ổn định tuyệt đối trên cả máy cục bộ lẫn Google Colab.
    """
    data_path = Path(data_dir)

    # Danh sách các thư mục ứng viên để tự động tìm kiếm (Local, Colab, Google Drive)
    candidate_dirs = [
        data_path,
        Path("..") / data_path,
        Path(".") / "Dataset",
        Path("..") / "Dataset",
        Path("./Dataset"),
        Path("../Dataset")
    ]

    dfile_path = None
    searched_paths = []

    for c_dir in candidate_dirs:
        for cand in [c_dir / f"{dtype}_data.txt", c_dir / f"{dtype}.txt"]:
            searched_paths.append(str(cand))
            if cand.exists():
                dfile_path = cand
                break
        if dfile_path is not None:
            break

    if dfile_path is None:
        raise FileNotFoundError(
            f"❌ Không tìm thấy tệp dữ liệu '{dtype}' trong các đường dẫn sau:\n" +
            "\n".join(f"  - {p}" for p in searched_paths[:6]) +
            f"\nVui lòng kiểm tra lại tham số --data_dir hoặc đảm bảo thư mục Dataset tồn tại!"
        )

    model_tag = str(tokenizer.name_or_path).replace('/', '_').replace('\\', '_')
    cached_path = dfile_path.with_name(f"{dfile_path.stem}_{model_tag}.cached")
    legacy_cached_path = dfile_path.with_suffix('.cached')

    if not overwrite_data and cached_path.exists():
        features = safe_torch_load(cached_path)
    elif not overwrite_data and legacy_cached_path.exists() and ('phobert-base' in model_tag):
        features = safe_torch_load(legacy_cached_path)
    else:
        features = convert_word_segment_examples_features(
            data_path=dfile_path,
            tokenizer=tokenizer,
            label2id=label2id,
            header_names=header,
            max_seq_len=max_seq_len
        )
        try:
            torch.save(features, cached_path)
        except Exception:
            pass

    return NerDataset(features)
