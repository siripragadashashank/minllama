import torch
from torch.optim import Optimizer

from typing import Callable, Iterable, Tuple


class AdamW(Optimizer):
    