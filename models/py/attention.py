import torch

from mimir_frontend.model_export import export


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Parameter(torch.randn(8, 8))
        self.wk = torch.nn.Parameter(torch.randn(8, 8))
        self.wv = torch.nn.Parameter(torch.randn(8, 8))
        self.wo = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        q = torch.ops.aten.mm.default(x, self.wq)
        k = torch.ops.aten.mm.default(x, self.wk)
        v = torch.ops.aten.mm.default(x, self.wv)
        scores = torch.ops.aten.mm.default(q, k.t())
        weights = torch.ops.aten.sigmoid.default(scores)
        context = torch.ops.aten.mm.default(weights, v)
        return torch.ops.aten.mm.default(context, self.wo)


export_to_mim = export(TinyAttention(), input_shapes=[(4, 8)], name="tiny_attention")


if __name__ == "__main__":
    from mimir_frontend.model_export import run_spec_with_mimir

    run_spec_with_mimir(export_to_mim)
