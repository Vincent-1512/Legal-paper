import os
from typing import Union, List
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


class NerFeatures(object):
    def __init__(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None, labels=None, label_masks=None, **kwargs):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.token_type_ids = (torch.as_tensor(token_type_ids, dtype=torch.long)
                               if token_type_ids is not None
                               else torch.zeros_like(self.input_ids))
        self.attention_mask = (torch.as_tensor(attention_mask, dtype=torch.long)
                               if attention_mask is not None
                               else torch.ones_like(self.input_ids))
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long) if valid_ids is not None else None
        self.labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long) if label_masks is not None else None


def convert_word_segment_examples_features(data_path: Union[str, os.PathLike],
                                           tokenizer,
                                           label2id: List[str],
                                           header_names: List[str],
                                           max_seq_len: int = 256) -> List[NerFeatures]:
    """
    Process Vietnamese word-segmented dataset into PhoBERT features with subword alignment.
    Aligns BPE subwords back to original words via valid_ids.
    Compatible with all pandas and PyTorch versions.
    """
    features = []
    tokens = []
    tag_ids = []

    # Đọc dữ liệu CoNLL (dấu cách hoặc tab)
    data = pd.read_csv(data_path,
                       sep=r'\s+',
                       encoding='utf-8',
                       skip_blank_lines=False,
                       names=header_names)
    
    # Hỗ trợ cả pandas 1.x và 2.x
    if hasattr(data, 'ffill'):
        data = data.ffill()
    else:
        data = data.fillna(method="ffill")

    total_rows = len(data)
    for row_idx, row in tqdm(data.iterrows(), total=total_rows, desc=f"Processing {os.path.basename(data_path)}..."):
        token_val = row.get('token', None) if hasattr(row, 'get') else getattr(row, 'token', None)
        ner_val = row.get('ner', None) if hasattr(row, 'get') else getattr(row, 'ner', None)

        if pd.notna(token_val) and str(token_val).strip() != '' and str(token_val).strip() != 'nan':
            tokens.append(str(token_val).strip().replace(' ', '_'))
            ner_tag = str(ner_val).strip()
            if ner_tag in label2id:
                tag_ids.append(label2id.index(ner_tag))
            else:
                tag_ids.append(0)  # default 'O'
            if row_idx < total_rows - 1:
                continue

        seq_len = len(tokens)
        if seq_len == 0:
            continue

        sentence = ' '.join(tokens)
        encoding = tokenizer(sentence,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len)

        subwords = tokenizer.tokenize(sentence)
        valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
        label_masks = np.zeros(len(encoding.input_ids), dtype=int)

        i = 1
        for idx, subword in enumerate(subwords[:max_seq_len - 2]):
            if idx != 0 and subwords[idx - 1].endswith("@@"):
                continue
            valid_ids[i - 1] = idx + 1
            i += 1

        if max_seq_len >= seq_len:
            label_padding_size = (max_seq_len - seq_len)
            label_masks[:seq_len] = [1] * seq_len
            tag_ids.extend([0] * label_padding_size)
        else:
            tag_ids = tag_ids[:max_seq_len]
            label_masks[:-2] = [1] * (max_seq_len - 2)
            tag_ids[-2:] = [0] * 2

        items = {key: val for key, val in encoding.items()}
        if 'token_type_ids' not in items:
            items['token_type_ids'] = [0] * max_seq_len
        items['labels'] = tag_ids
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_masks

        features.append(NerFeatures(**items))

        tokens = []
        tag_ids = []

    return features
