import torch

from mimir_frontend.model_export import export


class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0_w = torch.nn.Parameter(torch.randn(4, 1, 5, 5))
        self.conv0_b = torch.nn.Parameter(torch.randn(4))
        self.conv1_w = torch.nn.Parameter(torch.randn(6, 4, 5, 5))
        self.conv1_b = torch.nn.Parameter(torch.randn(6))
        self.fc_w = torch.nn.Parameter(torch.randn(6 * 5 * 5, 10))
        self.fc_b = torch.nn.Parameter(torch.randn(10))

    def forward(self, x):
        x = torch.ops.aten.convolution.default(
            x, self.conv0_w, self.conv0_b, [1, 1], [2, 2], [1, 1], False, [0, 0], 1
        )
        x = torch.ops.aten.relu.default(x)
        x = torch.ops.aten.max_pool2d.default(x, [2, 2], [2, 2], [0, 0], [1, 1])
        x = torch.ops.aten.convolution.default(
            x, self.conv1_w, self.conv1_b, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        x = torch.ops.aten.relu.default(x)
        x = torch.ops.aten.avg_pool2d.default(x, [2, 2], [2, 2], [0, 0], False, True, None)
        x = torch.ops.aten.view.default(x, [2, 6 * 5 * 5])
        return torch.ops.aten.addmm.default(self.fc_b, x, self.fc_w)


export_to_mim = export(LeNet(), input_shapes=[(2, 1, 28, 28)], name="lenet")


if __name__ == "__main__":
    from mimir_frontend.model_export import run_spec_with_mimir

    run_spec_with_mimir(export_to_mim)
