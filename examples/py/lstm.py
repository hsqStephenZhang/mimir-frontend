import os

import torch

import mimir_frontend.backend


class TinyLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wxi = torch.nn.Parameter(torch.randn(5, 7))
        self.whi = torch.nn.Parameter(torch.randn(7, 7))
        self.bi = torch.nn.Parameter(torch.randn(7))
        self.wxf = torch.nn.Parameter(torch.randn(5, 7))
        self.whf = torch.nn.Parameter(torch.randn(7, 7))
        self.bf = torch.nn.Parameter(torch.randn(7))
        self.wxg = torch.nn.Parameter(torch.randn(5, 7))
        self.whg = torch.nn.Parameter(torch.randn(7, 7))
        self.bg = torch.nn.Parameter(torch.randn(7))
        self.wxo = torch.nn.Parameter(torch.randn(5, 7))
        self.who = torch.nn.Parameter(torch.randn(7, 7))
        self.bo = torch.nn.Parameter(torch.randn(7))

    def gate(self, x_t, h, wx, wh, b):
        return torch.ops.aten.addmm.default(b, x_t, wx) + torch.ops.aten.mm.default(h, wh)

    def step(self, x_t, h, c):
        i = torch.ops.aten.sigmoid.default(self.gate(x_t, h, self.wxi, self.whi, self.bi))
        f = torch.ops.aten.sigmoid.default(self.gate(x_t, h, self.wxf, self.whf, self.bf))
        g = torch.ops.aten.tanh.default(self.gate(x_t, h, self.wxg, self.whg, self.bg))
        o = torch.ops.aten.sigmoid.default(self.gate(x_t, h, self.wxo, self.who, self.bo))
        c = f * c + i * g
        h = o * torch.ops.aten.tanh.default(c)
        return h, c

    @torch.compile(backend="mimir", options={"debug_dir": f"{os.path.dirname(os.path.realpath(__file__))}/../mim_debug"})
    def forward(self, x, h0, c0):
        h, c = self.step(x[0], h0, c0)
        h, c = self.step(x[1], h, c)
        h, c = self.step(x[2], h, c)
        return h


if __name__ == "__main__":
    model = TinyLSTM()
    x = torch.randn(3, 2, 5)
    h0 = torch.randn(2, 7)
    c0 = torch.randn(2, 7)
    with torch.no_grad():
        want = model(x, h0, c0)
        print("compiled output:", want)
