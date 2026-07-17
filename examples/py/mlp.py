import os

import torch

import mimir_frontend.backend

class ClassicMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(16, 32))
        self.b0 = torch.nn.Parameter(torch.randn(32))
        self.w1 = torch.nn.Parameter(torch.randn(32, 8))
        self.b1 = torch.nn.Parameter(torch.randn(8))

    @torch.compile(backend="mimir", options={"debug_dir": f"{os.path.dirname(os.path.realpath(__file__))}/../mim_debug"})
    def forward(self, x):
        x = torch.ops.aten.addmm.default(self.b0, x, self.w0)
        x = torch.ops.aten.relu.default(x)
        return torch.ops.aten.addmm.default(self.b1, x, self.w1)

if __name__ == "__main__":
    model = ClassicMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        want = model(x)
        print("compiled output:", want)    
