# Glacier.Tensor User Manual

## Contents

1. [Installation & Requirements](#1-installation--requirements)
2. [Tensors, Shapes & Unmanaged Views (`Glacier.Tensor.Core`)](#2-tensors-shapes--unmanaged-views-glaciertensorcore)
3. [Reverse-Mode Autograd (`Glacier.Tensor.Autograd`)](#3-reverse-mode-autograd-glaciertensorautograd)
4. [Layers, LoRA & Optimizers](#4-layers-lora--optimizers)
5. [Losses & Tiled Cross-Entropy](#5-losses--tiled-cross-entropy)
6. [CPU SIMD & GPU Acceleration (`Glacier.Tensor.Compute`)](#6-cpu-simd--gpu-acceleration-glaciertensorcompute)
7. [Polaris, Distributed AllReduce & Diagnostics](#7-polaris-distributed-allreduce--diagnostics)
8. [Building & Testing](#8-building--testing)

---

## 1. Installation & Requirements

```bash
dotnet add package Glacier.Tensor
```

- **Target framework:** `.NET 10` (`net10.0`), `<IsAotCompatible>true</IsAotCompatible>`, `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`.
- **Dependencies:** `Glacier.Gpu`, `Glacier.Compute`, and `Glacier.Polaris` (resolved cleanly via NuGet).

---

## 2. Tensors, Shapes & Unmanaged Views (`Glacier.Tensor.Core`)

`Tensor<T>` (`where T : unmanaged`) wraps a reference-counted `NativeMemoryBlock<T>` with up to 8 dimensions (`Shape8`, `Strides8`):

```csharp
using Glacier.Tensor.Core;

// Construction helpers
using var z = Tensor<float>.Zeros(4, 16);
using var fromArr = Tensor<float>.FromSpan([1f, 2f, 3f, 4f], [2, 2]);
using var ones = TensorFloatExtensions.Ones(2, 4);
using var norm = TensorFloatExtensions.RandomNormal([8, 16], mean: 0f, std: 0.02f, seed: 42);
using var kaiming = TensorFloatExtensions.KaimingUniform([16, 8], fanIn: 8, seed: 42);

// Zero-copy views (share underlying NativeMemoryBlock<T> via AddRef)
using var sliced = norm.Slice(dimension: 0, start: 2, length: 4);
using var transposed = sliced.Transpose();
using var reshaped = norm.Reshape(16, 8);
using var contig = transposed.Contiguous();
```

- Dispose tensors and views when done (`using var`) so `NativeMemoryBlock<T>` releases its unmanaged memory deterministically.
- Check `tensor.IsContiguous` before assuming flat linear layout across `tensor.AsSpan()` or `tensor.DataPointer`.
- Configure CPU worker parallelism with `TensorConcurrency.MaxDegreeOfParallelism`.

### Elementwise & Math Operations (`TensorOps`)

`TensorOps` records operations onto `AutogradTape.Current` when a tape is active:
- `Add`, `Subtract`, `Multiply`, `Scale`, `MatMul`
- `ReLU`, `Sigmoid`, `GELU`, `RMSNorm`, `Softmax(tensor, dim: -1)`

---

## 3. Reverse-Mode Autograd (`Glacier.Tensor.Autograd`)

`AutogradTape` tracks operations in a `[ThreadStatic]` stack (`AutogradTape.Current`), so inner nested `using var inner = new AutogradTape();` blocks restore outer tapes automatically upon disposal:

```csharp
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Losses;

using var w = TensorFloatExtensions.RandomNormal([4, 2], seed: 1);
using var x = TensorFloatExtensions.Ones(3, 4);
using var targets = Tensor<float>.Zeros(3, 2);

using var tape = new AutogradTape();
tape.Watch(w);

using var y = TensorOps.MatMul(x, w);
var (lossVal, lossTensor) = LossFunctions.MSELoss(y, targets);
using var _loss = lossTensor;

tape.Backward(lossTensor, retainGraph: false);
ReadOnlySpan<float> gradW = w.Grad!.AsSpan();
```

For manual buffer reuse in low-overhead transformer loops, rent unmanaged scratch memory from `AutogradScratchWorkspace.GetOrCreate(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim)` and free with `AutogradScratchWorkspace.DisposeAllThreadLocal()`.

---

## 4. Layers, LoRA & Optimizers

### Built-In Layers (`Glacier.Tensor.Layers`)

- **`Linear(inFeatures, outFeatures, seed)`:** Dense affine layer (`Weight`, `Bias`, `Forward(input)`, `ZeroGrad()`).
- **`LayerNorm(normalizedShape, eps = 1e-5f)`:** Layer normalization (`Gamma`, `Beta`, `Forward(input)`, `ZeroGrad()`).
- **`Sequential(params ILayer[] layers)`:** Composes multiple `ILayer` instances sequentially (`Parameters`, `Gradients`, `Forward`, `ZeroGrad`).
- **`LoraLinear`:** Wraps frozen base weights $W_0$ with low-rank trainable factors $A$ and $B$:

```csharp
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;

using var baseW = Tensor<float>.Zeros(1024, 1024);
using var lora = new LoraLinear(baseW, baseBias: null, rank: 16, alpha: 32f, seed: 7);
// Or allocate standalone: LoraLinear.Create(1024, 1024, rank: 16, alpha: 32f);

using var opt = new AdamW(lora.Parameters, lr: 1e-3f);
using var tape = new AutogradTape();
foreach (var p in lora.Parameters) tape.Watch(p);

using var pred = lora.Forward(TensorFloatExtensions.Ones(4, 1024));
var (_, loss) = LossFunctions.MSELoss(pred, Tensor<float>.Zeros(4, 1024));
using var _loss = loss;
tape.Backward(loss);
opt.Step();

// Fold (alpha / rank) * (A * B) into a standalone weight tensor
using var merged = lora.Merge();
```

### Optimizers (`Glacier.Tensor.Optimizers`)

- **`AdamW`** and **`SGD`** implement `IOptimizer` (`Step()`, `ZeroGrad()`, `Dispose()`).

---

## 5. Losses & Tiled Cross-Entropy

### `LossFunctions`

All five functions return `(float lossValue, Tensor<float> lossTensor)` compatible with `tape.Backward(lossTensor)`:

- `LossFunctions.MSELoss(predictions, targets)`
- `LossFunctions.BinaryCrossEntropy(predictions, targets)`
- `LossFunctions.CrossEntropy(logits, targets)`
- `LossFunctions.CategoricalCrossEntropyLoss(...)`
- `LossFunctions.BrierScoreLoss(...)` and `LossFunctions.ExpectedCalibrationError(...)`

### `FusedCrossEntropyLoss`

Computes online tiled Log-Sum-Exp cross-entropy (`VocabTileSize = 4096`) over large vocabulary matrices without allocating full `[seqLen, vocabSize]` logit tensors:

```csharp
float loss = FusedCrossEntropyLoss.ComputeLossAndGradients(
    finalHidden, targetTokens, lmHeadWeight); // and raw pointer overloads
```

---

## 6. CPU SIMD & GPU Acceleration (`Glacier.Tensor.Compute`)

### Target Selection (`GpuTarget` & `GpuAccelerator`)

Pass a `GpuTarget` to `TensorOps.MatMul(a, b, target)` or `GpuAccelerator.AcceleratedMatMul(a, b, c, target)`:

- `GpuTarget.Auto`, `GpuTarget.Cpu`, `GpuTarget.Nvidia`, `GpuTarget.NvidiaTensorCore`, `GpuTarget.Direct3D12`, `GpuTarget.Vulkan`, `GpuTarget.Amd`, `GpuTarget.DualGpu`.
- Query hardware availability at runtime with `GpuAccelerator.IsGpuAvailable`, `HasNvidiaGpu`, `HasAmdGpu`, `HasDirect3D12`, `HasVulkan`, and `CublasDriver.IsAvailable`.

### Persistent GPU Tensors (`DeviceResidentTensor<T>`)

Avoid repeated host-to-device and device-to-host copies in multi-step GPU loops:

```csharp
using Glacier.Tensor.Core;

using var aDev = DeviceResidentTensor<float>.FromHost(aHost);
using var bDev = DeviceResidentTensor<float>.FromHost(bHost);
using var cDev = DeviceResidentTensor<float>.Allocate(1024, 1024);

aDev.MatMul(bDev, cDev);
using Tensor<float> result = cDev.ToHost();
```

- Supports `AllocateCuda`, `AllocateD3D12`, `Upload`, `DownloadTo`, `View`, and `Slice`.
- On Windows Direct3D 12 errors (such as `DXGI_ERROR_DEVICE_REMOVED`), inspect `D3D12GemmKernel.LastError` (`D3D12KernelDiagnostics`, reporting `Stage`, `HResult`, dimensions, and `DeviceRemovedReasonDescription`) and clear thread-local errors with `D3D12GemmKernel.ClearLastError()`.
- Reclaim pooled CUDA contexts/scratch buffers with `CudaExecutionContext.TrimPool()` or `ClearPool()`.

---

## 7. Polaris, Distributed AllReduce & Diagnostics

- **Polaris Interop (`Glacier.Tensor.Interop.PolarisExtensions`):** Convert columns with `df.ToTensor("col1", "col2")`, `tensor.ToSeries("name")`, zero-copy `series.AsTensorView()`, and `df.ToTensorView(...)`.
- **Distributed AllReduce (`Glacier.Tensor.Distributed.GlacierDistributedTensorExtensions`):** Run `AllReduceAsync` / `AllReduceClusterAsync` across `Glacier.Compute` ring contexts directly on `Tensor<float>` buffers.
- **Diagnostics (`Glacier.Tensor.Diagnostics.GlacierDiagnostics`):** Route library logs via `ConsoleGlacierLogger`, `DelegateGlacierLogger`, or `NullGlacierLogger` (`LogLevel.Trace` through `Critical`).

---

## 8. Building & Testing

```bash
dotnet build Glacier.Tensor.slnx -c Release
dotnet test tests/Glacier.Tensor.Tests/Glacier.Tensor.Tests.csproj -c Release
```
