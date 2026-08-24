import os

import torch

import mimir_frontend.backend


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Parameter(torch.randn(8, 8))
        self.wk = torch.nn.Parameter(torch.randn(8, 8))
        self.wv = torch.nn.Parameter(torch.randn(8, 8))
        self.wo = torch.nn.Parameter(torch.randn(8, 8))

    @torch.compile(backend="mimir", options={"debug_dir": f"{os.path.dirname(os.path.realpath(__file__))}/../mim_debug"})
    def forward(self, x):
        q = torch.ops.aten.mm.default(x, self.wq)
        k = torch.ops.aten.mm.default(x, self.wk)
        v = torch.ops.aten.mm.default(x, self.wv)
        scores = torch.ops.aten.mm.default(q, k.t())
        weights = torch.ops.aten.sigmoid.default(scores)
        context = torch.ops.aten.mm.default(weights, v)
        return torch.ops.aten.mm.default(context, self.wo)


if __name__ == "__main__":
    model = TinyAttention()
    x = torch.randn(4, 8)
    with torch.no_grad():
        want = model(x)
        print("compiled output:", want)
