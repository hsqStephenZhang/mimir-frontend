# Vendored Lighthouse KernelBench fixtures

`ci.yaml`, `level1.yaml`, `level2.yaml`, and `level3.yaml` are verbatim copies of
`examples/KernelBench/*.yaml` from [libxsmm/lighthouse](https://github.com/libxsmm/lighthouse)
at commit `d330e7628298207aca4ce2d7d67277c50c867d85`, the revision pinned by
`.github/workflows/kernelbench.yml`. Lighthouse is licensed under the Apache
License v2.0 with LLVM Exceptions.

`scripts/run_kernelbench_mimir.py` reads these files when `--lighthouse` is not
given, and reads the checkout's own copies when it is. The sibling
`../level*.yaml` files are local overlays applied on top of either source.

The KernelBench sources themselves come from this repository's `KernelBench`
submodule, which pins the same KernelBench commit as Lighthouse
(`423217d9fda91e0c2d67e4a43bf62f96f6d104f1`).

To refresh after bumping the Lighthouse pin in the workflow, copy the four YAML
files from the new checkout and update the commit hash above.
