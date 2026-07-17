import torch

from mimir_frontend.model_export import export


class ClassicMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(16, 32))
        self.b0 = torch.nn.Parameter(torch.randn(32))
        self.w1 = torch.nn.Parameter(torch.randn(32, 8))
        self.b1 = torch.nn.Parameter(torch.randn(8))

    def forward(self, x):
        x = torch.ops.aten.addmm.default(self.b0, x, self.w0)
        x = torch.ops.aten.relu.default(x)
        return torch.ops.aten.addmm.default(self.b1, x, self.w1)


export_to_mim = export(ClassicMLP(), input_shapes=[(4, 16)], name="classic_mlp")


if __name__ == "__main__":
    from mimir_frontend.model_export import run_spec_with_mimir

    run_spec_with_mimir(export_to_mim)
