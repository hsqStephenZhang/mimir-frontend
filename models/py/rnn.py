import torch

from mimir_frontend.model_export import export


class TinyRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wx = torch.nn.Parameter(torch.randn(5, 7))
        self.wh = torch.nn.Parameter(torch.randn(7, 7))
        self.b = torch.nn.Parameter(torch.randn(7))

    def step(self, x_t, h):
        h = torch.ops.aten.addmm.default(self.b, x_t, self.wx) + torch.ops.aten.mm.default(h, self.wh)
        return torch.ops.aten.tanh.default(h)

    def forward(self, x, h0):
        h = self.step(x[0], h0)
        h = self.step(x[1], h)
        return self.step(x[2], h)


export_to_mim = export(TinyRNN(), input_shapes=[(3, 2, 5), (2, 7)], name="tiny_rnn")


if __name__ == "__main__":
    from mimir_frontend.model_export import run_spec_with_mimir

    run_spec_with_mimir(export_to_mim)
