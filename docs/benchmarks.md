# Glacier.Tensor Benchmarks

All measurements below come from physical hardware execution using the `Glacier.Tensor.Benchmarks` harness.

## Test Environment

- **CPU:** AMD Ryzen AI 9 HX 370 (12 cores / 24 threads, Zen 5, AVX-512)
- **Discrete GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6, `sm_89`)
- **RAM:** 31 GB
- **OS:** Windows 11 (`10.0.26300.0`)
- **Runtime:** `.NET 10.0.15` (`.NET SDK 11.0.100`, Release x64)

## Reproduce Locally

```bash
# Full console benchmark suite (CPU GEMM, Autograd MLP, LoRA layers, CUDA, D3D12, Vulkan)
dotnet run --project benchmarks/Glacier.Tensor.Benchmarks/Glacier.Tensor.Benchmarks.csproj -c Release

# BenchmarkDotNet GEMM suite
dotnet run --project benchmarks/Glacier.Tensor.Benchmarks/Glacier.Tensor.Benchmarks.csproj -c Release -- --bdn
```

---

## 1. CPU Cache-Blocked GEMM (`GemmKernels.MatMul`)

| Matrix Size | Average Latency | Throughput |
| :--- | ---: | ---: |
| `256 x 256` FP32 | `0.22 ms` | **154.47 GFLOPS** |
| `512 x 512` FP32 | `0.65 ms` | **413.18 GFLOPS** |
| `1024 x 1024` FP32 | `4.77 ms` | **449.80 GFLOPS** |
| `2048 x 2048` FP32 | `33.10 ms` | **519.07 GFLOPS** |

## 2. Autograd Training & CPU LoRA Projection Steps

| Workload | Scope | Avg Step Latency | Notes |
| :--- | :--- | ---: | :--- |
| **Autograd MLP Step** | Batch `64`, `128 -> 256 (LayerNorm+GELU) -> 16`, `CrossEntropy` + `AdamW` (50 steps) | **0.599 ms** | Backward managed alloc: `10,305 B/step` |
| **Qwen 2.5 7B `q_proj` LoRA Step** | `seq=512`, `3584 -> 3584`, `r=16`, `MSELoss` + `AdamW` | **21.94 ms** | `23,335 tok/s` |
| **Qwen 2.5 7B `gate_proj` LoRA Step** | `seq=512`, `3584 -> 18944`, `r=16`, `MSELoss` + `AdamW` | **154.19 ms** | `3,321 tok/s` |

## 3. GPU GEMM: Copied vs Device-Resident (NVIDIA RTX 4060 Laptop)

| Matrix Size | Backend & Mode | Latency | Throughput | Resident vs Copied |
| :--- | :--- | ---: | ---: | :---: |
| `1024 x 1024` | NVIDIA CUDA Copied (`ExecuteNvidiaGemm`) | `1.445 ms` | `1,486.4 GFLOPS` | — |
| `1024 x 1024` | NVIDIA CUDA Resident (`DeviceResidentTensor`) | `0.603 ms` | **3,561.3 GFLOPS (`3.56 TFLOPS`)** | `2.40x` |
| `2048 x 2048` | NVIDIA CUDA Copied (`ExecuteNvidiaGemm`) | `6.580 ms` | `2,611.0 GFLOPS` | — |
| `2048 x 2048` | NVIDIA CUDA Resident (`DeviceResidentTensor`) | `3.387 ms` | **5,073.0 GFLOPS (`5.07 TFLOPS`)** | `1.94x` |
| `1024 x 1024` | Direct3D 12 Copied (`D3D12GemmKernel.Execute`) | `2.178 ms` | `985.8 GFLOPS` | — |
| `1024 x 1024` | Direct3D 12 Resident (`ExecuteResident`) | `0.585 ms` | **3,671.3 GFLOPS (`3.67 TFLOPS`)** | `3.72x` |
| `2048 x 2048` | Direct3D 12 Copied (`D3D12GemmKernel.Execute`) | `9.466 ms` | `1,814.8 GFLOPS` | — |
| `2048 x 2048` | Direct3D 12 Resident (`ExecuteResident`) | `3.917 ms` | **4,386.4 GFLOPS (`4.39 TFLOPS`)** | `2.42x` |
| `1024 x 1024` | Vulkan 1.3 Copied (`VulkanGemmKernel.Execute`) | `95.809 ms` | `22.4 GFLOPS` | — |
| `2048 x 2048` | Vulkan 1.3 Copied (`VulkanGemmKernel.Execute`) | `1,031.468 ms` | `16.7 GFLOPS` | — |
