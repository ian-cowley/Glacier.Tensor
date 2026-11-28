# Glacier.Tensor Architecture

## 1. Memory Model (`Tensor<T>` & `NativeMemoryBlock<T>`)

```text
Tensor<T> (Shape8 dims, Strides8 strides, long ElementOffset, bool RequiresGrad)
  └── NativeMemoryBlock<T> (64-byte aligned unmanaged block, CAS refcount)
```

`NativeMemoryBlock<T>` manages raw unmanaged memory allocations (`NativeMemory.AlignedAlloc(..., 64)` or non-owning external wrappers) guarded by atomic `Interlocked.CompareExchange` reference counting (`AddRef()` / `Dispose()`) and finalizer safety. View operations (`Slice`, `Transpose`, `Reshape`) create a new `Tensor<T>` with adjusted `Shape8`, `Strides8`, and `ElementOffset`, incrementing the reference count on the shared `NativeMemoryBlock<T>` instead of copying data. `AsSpan(offset, length)` validates signed offsets and capacities to prevent integer overflow.

---

## 2. Reverse-Mode Autograd Engine

- **Scoped Tape Stack:** `AutogradTape` stores a `[ThreadStatic]` stack of active tapes so nested inner scopes isolate subgraphs and restore outer tapes when disposed.
- **Backward Traversal:** Each `TapeEntry` records an `AutogradOp` enum tag, input/output tensor references, and integer/scalar metadata. Backward passes walk the tape in reverse topological order and accumulate into `Tensor<T>.Grad` using SIMD FMA kernels (`TensorBackwardKernels.AccumulateScaledSimd`).
- **Scratch Workspaces:** `AutogradScratchWorkspace` pools thread-local unmanaged scratch slabs for transformer forward/backward intermediate tensors.

---

## 3. Compute & Dispatch Architecture

```text
TensorOps.MatMul / GpuAccelerator.AcceleratedMatMul(a, b, c, target)
  ├── GpuTarget.NvidiaTensorCore -> WMMA PTX Tensor Core kernel (sm_75+)
  ├── GpuTarget.Nvidia           -> FastGemmKernel PTX / CublasDriver SGEMM + CudaExecutionContext pool
  ├── GpuTarget.Direct3D12       -> D3D12GemmKernel (64x64 register-tiled HLSL compute shader)
  ├── GpuTarget.Vulkan           -> VulkanGemmKernel (embedded SPIR-V shader)
  └── GpuTarget.Cpu              -> GemmKernels.MatMul (AVX-512 8x16, AVX2 8x8, AdvSimd 4x4)
```

- **`CudaExecutionContext` Pool:** Rents thread-safe CUDA streams and reusable device staging buffers (`DevA`, `DevB`, `DevC`) to avoid per-call allocation and global driver locks.
- **`DeviceResidentTensor<T>`:** Keeps `GpuBuffer<T>` resident on the device across multiple `MatMul` calls, bypassing PCIe or staging copy overhead per operation.
- **`D3D12KernelDiagnostics`:** Captures thread-isolated execution stage (`D3D12ExecutionStage`), `HRESULT`, grid size, and DXGI `GetDeviceRemovedReason()` diagnostics if a display adapter resets.
