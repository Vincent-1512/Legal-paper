from typing import Optional, List, Tuple, Any
from collections import OrderedDict
import torch


class SupConOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = torch.FloatTensor([0.0])
    num_valid_anchors: int = 0

    def __init__(self, *args, **kwargs):
        super(SupConOutput, self).__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())


class LinearProbeOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = torch.FloatTensor([0.0])
    logits: Optional[torch.FloatTensor] = None
    tags: Optional[List[int]] = []

    def __init__(self, *args, **kwargs):
        super(LinearProbeOutput, self).__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())

