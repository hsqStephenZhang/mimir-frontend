from pathlib import Path
import subprocess
import sys

from mimir_frontend.model_export import export_spec_from_module, load_python_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_model_file_export_spec_convention():
    module = load_python_module(REPO_ROOT / "models" / "py" / "mlp.py")
    spec = export_spec_from_module(module)

    assert spec.name == "classic_mlp"
    assert spec.input_shapes == [(4, 16)]


def test_export_models_script_writes_mimir_files(tmp_path):
    model_file = tmp_path / "toy_model.py"
    model_file.write_text(
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, y):
        return x


def get_inputs():
    return [torch.rand(4, 16), torch.rand(4, 16)]


def get_init_inputs():
    return []


export_name = "toy_model"
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_models_to_mimir.py"),
            str(model_file),
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    out_file = tmp_path / "toy_model.mim"
    assert out_file.exists()
    assert "extern con toy_model" in out_file.read_text()
    assert "toy_model" in result.stdout
