# Changelog

## v1.1.0

- **Expanded Reverse-Mode Autograd & View Differentiation:** Added reverse-mode backpropagation through `Softmax` (arbitrary inner/outer axes), `LayerNorm` (`dInput`, `dGamma`, `dBeta`), strided tensor views (`Slice`, `Transpose`, `Reshape`, `Contiguous`), and all five `LossFunctions` (`MSELoss`, `BinaryCrossEntropy`, `CrossEntropy`, `CategoricalCrossEntropyLoss`, `BrierScoreLoss`).
- **Nested `[ThreadStatic]` Autograd Tape Stack:** Scoped `AutogradTape.Current` management so nested inner tapes restore outer tapes cleanly upon disposal.
- **Lock-Free CAS Reference Counting & Finalizer Safety (`NativeMemoryBlock<T>`):** Atomic compare-exchange reference counting prevents `AddRef`/`Dispose` races, plus automatic intermediate activation disposal in `LoraLinear` and `Sequential` during inference.
- **Multi-Architecture CUDA Context Binding & Native AOT Hardening:** Added thread-safe worker context binding in `CudaExecutionContext` and `DeviceResidentTensor<T>`, Turing+ PTX headers (`sm_75+`), isolated Windows Direct3D 12 initialization, `<IsAotCompatible>true</IsAotCompatible>`, and NuGet OIDC publishing.

## v1.0.8

- **Unmanaged Memory Bounds Validation (`NativeMemoryBlock<T>`):** Added strict bounds checking in `AsSpan(offset, length)` against negative indices and overflow.
- **Thread-Isolated Direct3D 12 Diagnostics:** Added `D3D12KernelDiagnostics` (`D3D12ExecutionStage`, `GetDeviceRemovedReason`) backed by `[ThreadStatic]` error state.
- **Ambient Logging:** Integrated `GlacierDiagnostics` / `IGlacierLogger`.

## v1.0.7

- **`CudaExecutionContext` Pool:** Replaced the global initialization lock with a lock-free reusable CUDA context pool.
- **SIMD Tape Gradient Accumulation:** Added `Vector256<float>` FMA gradient accumulation into unmanaged parameter buffers.
