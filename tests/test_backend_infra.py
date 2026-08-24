from pathlib import Path

import pytest

import mimir_frontend.backend as backend


@pytest.mark.parametrize(
    ("system", "suffix"),
    [("Linux", ".so"), ("Darwin", ".dylib"), ("Windows", ".dll")],
)
def test_shared_library_suffix_is_platform_specific(system, suffix, monkeypatch):
    monkeypatch.setattr(backend.platform, "system", lambda: system)
    assert backend._shared_lib_suffix() == suffix


def test_invalid_cached_library_is_removed(tmp_path: Path, monkeypatch):
    cached_library = tmp_path / f"broken{backend._shared_lib_suffix()}"
    cached_library.write_bytes(b"not a shared library")

    def reject(_):
        raise OSError("invalid shared library")

    monkeypatch.setattr(backend.ctypes.cdll, "LoadLibrary", reject)

    assert backend._load_cached_library(cached_library) is None
    assert not cached_library.exists()
