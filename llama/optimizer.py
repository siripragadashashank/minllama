import torch
from torch.optim import Optimizer

from typing import Callable, Iterable, Tuple


class AdamW(Optimizer):
    def __init__(self,
                 params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.99),
                 eps: float = 1e-6,
                 weight_decay: float = 0.0,
                 correct_bias: bool = True
                 ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        pass

