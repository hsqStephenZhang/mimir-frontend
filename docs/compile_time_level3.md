# KernelBench level 3 compile times (full size)

Raw tables: `compile_time_level3_m2pro.csv`, `compile_time_level3_i5-14600kf.csv`.

Measured with `scripts/run_kernelbench_mimir.py --suite level3 --size-divisor 1 --max-fp-iters 8192`
under `MIMIR_PROFILE=summary MIMIR_DEBUG_DIR=...`, summarized by `scripts/summarize_compile_times.py`.
Every case runs in its own subprocess with a cold JIT cache, so each number is one cold compile of the
whole model (one Dynamo graph, `fullgraph=True`). 45 of the 50 models compile; the remaining five
(32, 33, 35, 48, 49) hit unsupported constructs and are not timed.

Columns, in seconds:

- `dynamo`: `torch.compile` tracing and guard setup, measured as first call minus backend minus a cached execution.
- `translate`: decomposition policy plus FX -> MimIR term construction through the Python bindings.
- `mimir`: all MimIR phases (`world.optimize()`) except LLVM IR emission - typing, partial evaluation,
  torch/tensor lowering, bufferisation, store elimination (`mem.seo`), scalarisation.
- `ll_emit`: the `runtime.ll_emit` phase that prints LLVM IR text.
- `clang`: `clang -O3 -march=native` on the emitted `.ll` producing the shared library, plus `dlopen`.
- `backend`: the whole backend call (`translate + mimir + ll_emit + clang` and bookkeeping).
- `pre_llvm`: `dynamo + translate + mimir`, i.e. everything before LLVM IR exists.

## Apple M2 Pro (macOS, Homebrew clang), 2026-10-30

MimIR fork `frontend-dev` at `9d2bf7f6a` (scoped `add_mem` splice), frontend `01df29b`.

| model | dynamo | translate | mimir | ll_emit | clang | backend | pre_llvm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1_MLP | 0.14 | 0.00 | 1.21 | 0.00 | 0.34 | 1.80 | 1.35 |
| 2_ShallowWideMLP | 1.38 | 0.00 | 1.36 | 0.01 | 0.38 | 2.22 | 2.74 |
| 3_DeepNarrowMLP | 0.17 | 0.00 | 1.40 | 0.02 | 0.47 | 2.18 | 1.57 |
| 4_LeNet5 | 0.15 | 0.00 | 1.29 | 0.01 | 0.68 | 2.24 | 1.44 |
| 5_AlexNet | 0.19 | 0.00 | 1.34 | 0.01 | 0.56 | 2.18 | 1.54 |
| 6_GoogleNetInceptionModule | 0.16 | 0.00 | 1.25 | 0.01 | 0.80 | 2.32 | 1.42 |
| 7_GoogleNetInceptionV1 | 0.38 | 0.01 | 2.77 | 0.08 | 2.11 | 5.52 | 3.17 |
| 8_ResNetBasicBlock | 0.17 | 0.00 | 1.30 | 0.01 | 0.70 | 2.27 | 1.47 |
| 9_ResNet18 | 0.33 | 0.00 | 2.13 | 0.06 | 1.97 | 4.65 | 2.47 |
| 10_ResNet101 | 0.34 | 0.01 | 11.90 | 0.30 | 56.28 | 71.95 | 12.26 |
| 11_VGG16 | 0.26 | 0.00 | 1.48 | 0.02 | 0.92 | 2.74 | 1.75 |
| 12_VGG19 | 0.25 | 0.00 | 1.54 | 0.03 | 0.97 | 2.86 | 1.79 |
| 13_DenseNet121TransitionLayer | 0.17 | 0.00 | 1.23 | 0.00 | 0.43 | 1.93 | 1.40 |
| 14_DenseNet121DenseBlock | 0.17 | 0.00 | 1.24 | 0.01 | 0.79 | 2.30 | 1.42 |
| 15_DenseNet121 | 0.66 | 0.01 | 8.37 | 0.22 | 34.72 | 45.54 | 9.05 |
| 16_DenseNet201 | 1.27 | 0.03 | 21.46 | 0.41 | 85.95 | 113.67 | 22.75 |
| 17_SqueezeNetFireModule | 0.17 | 0.00 | 1.25 | 0.00 | 0.52 | 2.04 | 1.42 |
| 18_SqueezeNet | 0.28 | 0.00 | 1.72 | 0.03 | 1.40 | 3.50 | 2.00 |
| 19_MobileNetV1 | 0.42 | 0.00 | 2.45 | 0.07 | 4.79 | 7.87 | 2.87 |
| 20_MobileNetV2 | 0.69 | 0.01 | 4.53 | 0.15 | 28.19 | 34.27 | 5.23 |
| 21_EfficientNetMBConv | 0.18 | 0.00 | 1.25 | 0.01 | 0.58 | 2.11 | 1.44 |
| 22_EfficientNetB0 | 0.55 | 0.01 | 3.30 | 0.10 | 21.43 | 25.75 | 3.85 |
| 23_EfficientNetB1 | 0.40 | 0.00 | 2.20 | 0.06 | 14.20 | 16.98 | 2.61 |
| 24_EfficientNetB2 | 0.39 | 0.00 | 1.93 | 0.05 | 2.06 | 4.45 | 2.32 |
| 25_ShuffleNetUnit | 0.19 | 0.00 | 1.29 | 0.01 | 0.52 | 2.10 | 1.49 |
| 26_ShuffleNet | 0.31 | 0.00 | 1.74 | 0.04 | 1.65 | 3.80 | 2.05 |
| 27_RegNet | 0.23 | 0.00 | 1.53 | 0.01 | 0.63 | 2.45 | 1.76 |
| 28_VisionTransformer | 0.29 | 0.13 | 3.61 | 0.11 | 9.12 | 14.08 | 4.02 |
| 29_SwinMLP | 0.29 | 0.00 | 1.64 | 0.03 | 1.29 | 3.34 | 1.94 |
| 30_SwinTransformerV2 | 0.65 | 0.02 | 11.32 | 0.30 | 118.84 | 134.38 | 11.98 |
| 31_VisionAttention | 0.18 | 0.02 | 1.30 | 0.01 | 0.65 | 2.24 | 1.49 |
| 34_VanillaRNNHidden | 0.15 | 0.00 | 1.25 | 0.01 | 0.64 | 2.16 | 1.40 |
| 36_LSTMHn | 0.22 | 0.00 | 1.54 | 0.02 | 17.76 | 19.62 | 1.76 |
| 37_LSTMCn | 0.22 | 0.00 | 1.59 | 0.02 | 18.36 | 20.27 | 1.81 |
| 38_LSTMBidirectional | 0.28 | 0.00 | 2.08 | 0.05 | 44.32 | 46.81 | 2.37 |
| 39_GRU | 4.07 | 0.00 | 1.63 | 0.02 | 23.75 | 25.70 | 5.70 |
| 40_GRUHidden | 4.20 | 0.00 | 1.76 | 0.02 | 23.12 | 25.29 | 5.95 |
| 41_GRUBidirectional | 7.47 | 0.00 | 2.06 | 0.05 | 50.53 | 52.99 | 9.53 |
| 42_GRUBidirectionalHidden | 7.54 | 0.00 | 2.14 | 0.05 | 45.44 | 47.98 | 9.69 |
| 43_MinGPTCausalAttention | 0.16 | 0.00 | 1.30 | 0.01 | 0.95 | 2.53 | 1.47 |
| 44_MiniGPTBlock | 0.19 | 0.00 | 1.44 | 0.02 | 1.58 | 3.33 | 1.64 |
| 45_UNetSoftmax | 0.36 | 0.00 | 2.61 | 0.08 | 4.83 | 8.14 | 2.97 |
| 46_NetVladWithGhostClusters | 0.16 | 0.00 | 1.27 | 0.01 | 0.51 | 2.05 | 1.43 |
| 47_NetVladNoGhostClusters | 0.16 | 0.00 | 1.25 | 0.01 | 0.44 | 1.95 | 1.42 |
| 50_ReLUSelfAttention | 0.14 | 0.00 | 1.60 | 0.01 | 0.77 | 2.64 | 1.75 |

Geomean over the 45 compiled models: pre-LLVM 2.57 s, backend 6.60 s; `clang` alone 2.91 s.
Summed over all 45 models the split is dynamo 4.5 %, translate 0.0 %, MimIR phases 15.3 %,
LLVM IR emission 0.3 %, clang 76.1 %. The pre-LLVM share is 19.8 %.

Observations:

- Even the smallest model pays about 1.2 s of MimIR phases: the pipeline runs over the whole torch
  dialect (`torch.mim`, about 250 helper lambdas) before specialising to the graph.
- MimIR time grows with graph size mainly through `SEO::Analyzer` (store elimination rounds, about
  six per FX node): DenseNet201 spends 12.6 s of its 21.5 s there, ResNet101 and SwinV2 about half.
- clang dominates the deep models (DenseNet201 86 s, SwinTransformerV2 119 s, ResNet101 56 s) and the
  recurrent ones (bidirectional GRU/LSTM 45-50 s), because the emitted IR unrolls every layer into its
  own loop nest.

## Intel i5-14600KF (Manjaro Linux, gcc 15 for MimIR, clang 22 for the JIT), 2026-10-30

Same commits, 20 threads, 62 GB. Raw table: `compile_time_level3_i5-14600kf.csv`.

| model | dynamo | translate | mimir | ll_emit | clang | backend | pre_llvm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1_MLP | 0.17 | 0.00 | 1.28 | 0.00 | 0.07 | 1.59 | 1.45 |
| 2_ShallowWideMLP | 0.32 | 0.00 | 1.29 | 0.00 | 0.07 | 1.60 | 1.62 |
| 3_DeepNarrowMLP | 0.11 | 0.00 | 1.46 | 0.02 | 0.30 | 2.04 | 1.57 |
| 4_LeNet5 | 0.19 | 0.00 | 1.39 | 0.01 | 0.37 | 2.01 | 1.58 |
| 5_AlexNet | 0.26 | 0.00 | 1.45 | 0.01 | 0.37 | 2.08 | 1.72 |
| 6_GoogleNetInceptionModule | 0.19 | 0.00 | 1.36 | 0.01 | 0.24 | 1.85 | 1.55 |
| 7_GoogleNetInceptionV1 | 0.46 | 0.01 | 2.91 | 0.08 | 2.16 | 5.55 | 3.37 |
| 8_ResNetBasicBlock | 0.20 | 0.00 | 1.37 | 0.01 | 0.50 | 2.13 | 1.57 |
| 9_ResNet18 | 0.43 | 0.00 | 2.19 | 0.06 | 1.98 | 4.58 | 2.62 |
| 10_ResNet101 | 0.59 | 0.01 | 11.37 | 0.34 | 158.11 | 171.65 | 11.97 |
| 11_VGG16 | 0.32 | 0.00 | 1.57 | 0.02 | 0.70 | 2.56 | 1.89 |
| 12_VGG19 | 0.21 | 0.00 | 1.62 | 0.02 | 0.83 | 2.74 | 1.83 |
| 13_DenseNet121TransitionLayer | 0.20 | 0.00 | 1.33 | 0.00 | 0.12 | 1.69 | 1.53 |
| 14_DenseNet121DenseBlock | 0.21 | 0.00 | 1.33 | 0.00 | 0.18 | 1.76 | 1.55 |
| 15_DenseNet121 | 0.93 | 0.01 | 7.71 | 0.40 | 75.19 | 84.38 | 8.66 |
| 16_DenseNet201 | 1.44 | 0.02 | 18.34 | 0.47 | 176.89 | 198.20 | 19.81 |
| 17_SqueezeNetFireModule | 0.19 | 0.00 | 1.30 | 0.00 | 0.27 | 1.82 | 1.49 |
| 18_SqueezeNet | 0.35 | 0.01 | 1.78 | 0.03 | 1.28 | 3.39 | 2.14 |
| 19_MobileNetV1 | 0.49 | 0.00 | 2.38 | 0.07 | 5.94 | 8.77 | 2.87 |
| 20_MobileNetV2 | 0.52 | 0.01 | 4.05 | 0.15 | 62.17 | 67.03 | 4.57 |
| 21_EfficientNetMBConv | 0.22 | 0.00 | 1.35 | 0.01 | 0.20 | 1.81 | 1.57 |
| 22_EfficientNetB0 | 0.54 | 0.01 | 3.18 | 0.11 | 46.60 | 50.40 | 3.72 |
| 23_EfficientNetB1 | 0.42 | 0.00 | 2.26 | 0.06 | 22.19 | 24.88 | 2.69 |
| 24_EfficientNetB2 | 0.46 | 0.00 | 2.01 | 0.05 | 1.52 | 3.90 | 2.47 |
| 25_ShuffleNetUnit | 0.21 | 0.00 | 1.39 | 0.01 | 0.33 | 1.98 | 1.60 |
| 26_ShuffleNet | 0.35 | 0.00 | 1.82 | 0.04 | 1.46 | 3.62 | 2.18 |
| 27_RegNet | 0.27 | 0.00 | 1.44 | 0.01 | 0.36 | 2.06 | 1.71 |
| 28_VisionTransformer | 0.39 | 0.15 | 3.11 | 0.12 | 12.88 | 16.84 | 3.65 |
| 29_SwinMLP | 0.36 | 0.00 | 1.66 | 0.02 | 0.83 | 2.80 | 2.03 |
| 30_SwinTransformerV2 | 0.73 | 0.02 | 9.75 | 0.33 | 175.01 | 186.99 | 10.50 |
| 31_VisionAttention | 0.21 | 0.02 | 1.39 | 0.01 | 0.45 | 2.11 | 1.61 |
| 34_VanillaRNNHidden | 0.19 | 0.00 | 1.34 | 0.00 | 0.20 | 1.78 | 1.53 |
| 36_LSTMHn | 0.11 | 0.00 | 1.62 | 0.02 | 13.50 | 15.39 | 1.73 |
| 37_LSTMCn | 0.13 | 0.00 | 1.62 | 0.02 | 13.65 | 15.55 | 1.75 |
| 38_LSTMBidirectional | 0.16 | 0.00 | 2.15 | 0.05 | 36.29 | 38.76 | 2.31 |
| 39_GRU | 4.23 | 0.00 | 1.58 | 0.02 | 14.88 | 16.74 | 5.81 |
| 40_GRUHidden | 4.30 | 0.00 | 1.64 | 0.02 | 13.13 | 15.05 | 5.95 |
| 41_GRUBidirectional | 7.78 | 0.00 | 2.16 | 0.05 | 33.97 | 36.46 | 9.94 |
| 42_GRUBidirectionalHidden | 7.96 | 0.00 | 2.21 | 0.05 | 30.51 | 33.05 | 10.17 |
| 43_MinGPTCausalAttention | 0.20 | 0.00 | 1.40 | 0.01 | 0.43 | 2.09 | 1.60 |
| 44_MiniGPTBlock | 0.22 | 0.00 | 1.50 | 0.01 | 1.14 | 2.91 | 1.71 |
| 45_UNetSoftmax | 0.44 | 0.00 | 2.60 | 0.08 | 5.76 | 8.86 | 3.04 |
| 46_NetVladWithGhostClusters | 0.19 | 0.00 | 1.35 | 0.00 | 0.20 | 1.80 | 1.54 |
| 47_NetVladNoGhostClusters | 0.20 | 0.00 | 1.35 | 0.00 | 0.18 | 1.78 | 1.55 |
| 50_ReLUSelfAttention | 0.18 | 0.00 | 1.35 | 0.00 | 0.31 | 1.92 | 1.53 |

Geomean over the 45 compiled models: pre-LLVM 2.60 s, backend 6.40 s; `clang` alone 2.06 s.
Summed over all models: dynamo 3.5 %, translate 0.0 %, MimIR phases 11.0 %, LLVM IR emission 0.3 %,
clang 83.4 %. The pre-LLVM share is 14.5 %.

The MimIR phase times agree with the M2 Pro within about 15 % on every model, while clang 22 on x86 is
roughly twice as slow as Homebrew clang on the M2 Pro for the deep models (DenseNet201 177 s vs 86 s,
ResNet101 158 s vs 56 s).

## Reading for the rebuttal

- The pre-LLVM MimIR pipeline (type checking, partial evaluation, lowering, bufferisation, store
  elimination) is 2.0 s geomean on both machines; the geomean is dominated by the 1.2 s fixed cost.
- End to end, MimIR -> LLVM IR -> shared library is 6.4-6.6 s geomean, of which clang is 2-3 s geomean
  and 76-83 % of the summed time.
- Where MimIR is slow on large graphs the cause is `SEO::Analyzer` (store elimination's sparse
  fixed point), not type checking or partial evaluation: DenseNet201 12.6 s of its 21.5 s on the M2 Pro, SwinV2 and
  ResNet101 about half of their MimIR time.
- The MimIR -> MLIR path is not measured here: it needs the external `mim_mlir` plugin
  (HarithF/mim_mlir), which targets upstream MimIR master and has not been built against this fork.
