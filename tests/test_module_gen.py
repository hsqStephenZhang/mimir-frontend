import torch
import mim
from mimir_frontend.utils import model_to_mimir

def test_mimir_module_generation():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(10, 20))
            self.bias = torch.nn.Parameter(torch.randn(10))

        def forward(self, x):
            return torch.relu(torch.mm(x, self.weight.t()) + self.bias)

    model = SimpleModel()
    # Dynamic batch size 'n'
    input_shapes = [("n", 20)]
    
    mimir_ir = model_to_mimir(model, input_shapes, name="my_module", compile_phase="high_level")
    
    print("\nGenerated MimIR Module:")
    print(mimir_ir)
    
    assert "extern con my_module" in mimir_ir
    # Check if a Nat parameter is present
    assert "Nat" in mimir_ir

if __name__ == "__main__":
    test_mimir_module_generation()
