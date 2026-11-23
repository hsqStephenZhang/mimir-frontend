# Rebuttal drafts: benchmark coverage and static-shape scope

Two checklist items. English text is meant to be pasted; `[[VERIFY: …]]` marks a fact only the authors can
supply, `[[CZ: …]]` a decision. Numbers were measured on 2026-11-23 unless a date is given.

## Item 1 (A, D): coverage, failure reasons, tool versions, CPU reference pipeline, models added

### Draft

> **Coverage.** KernelBench Level 3 has 50 models. Figure 4 reports a selection of 25 of them, those we ran
> through all three CPU pipelines; the selection was not limited by MimIR. Rebuilt at the submission revision
> (frontend `2f27ccd`, MimIR `3a31afcde`), **44 of the 50 models compile at full size** and match PyTorch eager
> execution (`rtol = atol = 1e-4`); with the current revision it is **45 of 50**, the difference being
> SwinTransformerV2, whose LLVM IR the runtime emitter used to rejects as a redefinition (fixed since). All 25
> models of Figure 4 are among them. The revision extends Figure 4 by [[CZ: the 7 models to add]] of the other
> 20, those for which the reference pipeline also runs.
>
> **Why the other 5 fail.** All five fail in the Python front end, before MimIR sees a program, because a
> PyTorch target has no conversion yet: 32 ConvolutionalVisionTransformer uses the fused
> `_transformer_encoder_layer_fwd`; 33 VanillaRNN updates a tensor in place with `copy_`; 35 LSTM calls
> `torch.randn` inside the model; 48 and 49 Mamba2 use einops' `rearrange`. None is a limitation of the IR or
> of the lowering, and we list them in the artifact.
>
> **CPU reference pipeline.** PyTorch → Torch-MLIR [[VERIFY: version/commit]] → `linalg`-on-tensors, lowered
> with upstream MLIR passes [[VERIFY: LLVM/MLIR commit]] and a schedule that matches convolutions and
> contractions and applies interchange and tiling → LLVM IR → `opt -O3`, `llc -O3 -mcpu=native`. The MimIR
> pipelines use the same LLVM tools on the emitted IR. Single thread, pinned to one performance core, median of
> 10 runs, Intel Core Ultra 7 265H.
>
> **Tool versions.** MimIR `frontend-dev` @ `ab36445e8` on upstream `22295693d` [[VERIFY: commit used for the
> paper's numbers]]; PyTorch 2.12.0; Python 3.14.8; Homebrew clang 22.1.4 (M2 Pro) and clang 22.1.3 (Linux);
> gcc 15 to build MimIR on CI [[VERIFY: toolchain on the Ultra 7 machine]].

### Facts behind it

| Fact | Value | Where |
|---|---|---|
| Models in KernelBench Level 3 | 50 | `KernelBench/level3` |
| Compile and match eager, full size, now | 45 (90 %) | CI run 42060893472 (macOS), `docs/compile_time_level3.md` |
| Same, at the submission revision (`feat/torch-plugin-new`) | 44 (88 %); only SwinTransformerV2 differs | rebuilt 2026-11-24, MimIR `3a31afcde`, torch plugin `d93b656`, runtime plugin `6dee9c5` |
| Linux CI without memory-hungry models | 42/42 (ResNet101, DenseNet201, SwinV2 excluded by the 15 GB limit, not by failure) | same run |
| Models in Figure 4 | 25, all among the 45 | paper Fig. 4 caption |
| Compile but not in Figure 4 | 20 | list below |
| Failing models | 5, front-end conversions only | table below |

Failures, one line each (reproduced 2026-11-23 at full size):

| Model | Error raised by the front end |
|---|---|
| 32 ConvolutionalVisionTransformer | `torch._transformer_encoder_layer_fwd` not supported |
| 33 VanillaRNN | `copy_` not supported |
| 35 LSTM | `torch.randn` not supported |
| 48 Mamba2ReturnY, 49 Mamba2ReturnFinalState | einops `rearrange` not supported |

The 20 models that compile but are not in Figure 4: DenseNet201, SqueezeNetFireModule, SqueezeNet,
EfficientNetB2, VisionTransformer, SwinMLP, SwinTransformerV2, VisionAttention, VanillaRNNHidden, LSTMHn,
LSTMCn, LSTMBidirectional, GRU, GRUHidden, GRUBidirectional, GRUBidirectionalHidden, MinGPTCausalAttention,
MiniGPTBlock, UNetSoftmax, ReLUSelfAttention.

### Open point: "the 7 models that now compile"

The submission revision already compiles 44 models, so the 7 are **not** newly compilable models: they can only
be models *added to Figure 4*. Which 7 depends on where the Torch-MLIR reference pipeline runs, which this
repository cannot check; the pool is the 20 models above. Measured with the current harness, `--max-fp-iters 8192`
(the limit that large models such as ResNet101 and DenseNet201 need; the submission-era default was lower),
full size, M2 Pro; per-model detail in `/private/tmp/…/baseline-out/level3.json` of the session that produced it.

## Item 2 (C, D): scope the claims to static shapes

### Draft

> **Scope.** All measurements in Section 8 use static shapes: every extent is a literal when MimIR compiles the
> model. We will say so in the abstract, in Section 8, and in the conclusion, and move dynamic shapes to future
> work.
>
> **What Sections 6 and 7 do claim.** That an extent is a term, so a shape may mention a function parameter, the
> relations among extents stay visible to the type checker, and the tiling library applies unchanged to such
> shapes. That is a statement about the representation, and the paper gives no performance numbers for it.
>
> **What exists beyond the paper.** Since MimIR's runtime-extent bufferization landed upstream (mimir#646,
> 2026-11-16), a function with a symbolic extent compiles to LLVM and runs. We drove it from PyTorch with
> `torch.compile(dynamic=True)`: 22 model/shape combinations (MLP, convolution + batch norm + pooling,
> layer norm, softmax, batched matmul, `cat` along a dynamic axis, multiple outputs, …) each compile **once** and
> match eager for three sizes. Two attention cases do not yet compile, because the generic rank-5
> `map_reduce` schedule leaves a higher-order function behind when an extent is only known at runtime. We did
> not measure performance on dynamic shapes and make no claim about it.

### Facts behind it

* Static-only evaluation: the paper's Section 8 compiles fixed-shape models (`get_inputs()` of KernelBench),
  CPU pipelines (a), (b), (c) and the GPU pipeline all on those.
* Symbolic text in the paper, to keep but not extend: Section 6 "Tensor Shapes" and "Symbolic and runtime
  extents"; Section 7 `tensor.if_static` and the `k·4` example.
* End-to-end dynamic tests: `tests/test_dynamic_shapes.py` (22 model/shape cases passed, 2 `xfail(strict)`, 4
  helper tests), one graph per case (`unique_graphs == 1`) for sizes 5, 3 and 7.
* MimIR side: mimir#646 (runtime extents, merged 2026-11-16); mimir#660 (concat along a runtime extent, draft,
  with the executable test `lit/tensor/concat_dynamic.mim`).
* Not covered: performance on dynamic shapes; generic `map_reduce` (3-D `@` 3-D matmul, attention) with a runtime
  extent; any model with a runtime extent through the GPU pipeline.

### Suggested edits to the paper

1. Abstract and introduction: "static-shape" before "tensor programs" wherever performance is claimed.
2. Section 8 first paragraph: one sentence, "All models are compiled for static shapes."
3. Section 6, end of "Symbolic and runtime extents": add "We do not evaluate runtime extents here."
4. Conclusion: dynamic shapes as future work, naming the generic `map_reduce` schedule as the known gap.
