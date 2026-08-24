import os

import torch

import mimir_frontend.backend


class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0_w = torch.nn.Parameter(torch.randn(4, 1, 5, 5))
        self.conv0_b = torch.nn.Parameter(torch.randn(4))
        self.conv1_w = torch.nn.Parameter(torch.randn(6, 4, 5, 5))
        self.conv1_b = torch.nn.Parameter(torch.randn(6))
        self.fc_w = torch.nn.Parameter(torch.randn(6 * 5 * 5, 10))
        self.fc_b = torch.nn.Parameter(torch.randn(10))

    @torch.compile(backend="mimir", options={"debug_dir": f"{os.path.dirname(os.path.realpath(__file__))}/../mim_debug"})
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

if __name__ == "__main__":
    model = LeNet()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        want = model(x)
        print("compiled output:", want)
