import os

import torch

import mimir_frontend.backend

class TinyRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wx = torch.nn.Parameter(torch.randn(5, 7))
        self.wh = torch.nn.Parameter(torch.randn(7, 7))
        self.b = torch.nn.Parameter(torch.randn(7))

    def step(self, x_t, h):
        h = torch.ops.aten.addmm.default(self.b, x_t, self.wx) + torch.ops.aten.mm.default(h, self.wh)
        return torch.ops.aten.tanh.default(h)

    @torch.compile(backend="mimir", options={"debug_dir": f"{os.path.dirname(os.path.realpath(__file__))}/../mim_debug"})
    def forward(self, x, h0):
        h = self.step(x[0], h0)
        h = self.step(x[1], h)
        return self.step(x[2], h)

if __name__ == "__main__":
    model = TinyRNN()
    x = torch.randn(3, 2, 5)
    h0 = torch.randn(2, 7)
    with torch.no_grad():
        want = model(x, h0)
        print("compiled output:", want)
