import torch
import unittest
import numpy as np

from llama.optimizer import AdamW

seed = 0


class TestAdamW(unittest.TestCase):

    def setUp(self):
        self.ref = torch.tensor(np.load("optimizer_test.npy"))

    def tearDown(self):
        self.ref = None

    def test_optimizer(self, optimizer=AdamW):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        model = torch.nn.Linear(3, 2, bias=False)
        opt = optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            correct_bias=True
        )

        for i in range(10):
            opt.zero_grad()
            x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
            y_pred = model(x)
            y_actual = torch.tensor([x[0] + x[1], -x[2]])
            loss = ((y_pred - y_actual) ** 2).sum()
            loss.backward()
            opt.step()

        out = model.weight.detach()
        assert torch.allclose(self.ref, out)



