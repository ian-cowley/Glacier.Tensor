namespace Glacier.Tensor.Compute;

/// <summary>
/// Specifies the hardware target for matrix and tensor computations.
/// </summary>
public enum GpuTarget
{
    /// <summary>
    /// Intelligently dispatches between multi-threaded AVX-512 CPU, NVIDIA dGPU, and AMD APU
    /// based on matrix dimensions, memory capacity, and hardware availability.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Bare-metal discrete NVIDIA GPU execution via direct driver API (nvcuda.dll) and SASS machine code.
    /// Peak sustained TFLOPS for large matrix operations.
    /// </summary>
    Nvidia = 1,

    /// <summary>
    /// Bare-metal integrated AMD Radeon APU execution via direct driver API (amdhip64.dll)
    /// leveraging zero-copy unified system memory (LPDDR5X).
    /// </summary>
    Amd = 2,

    /// <summary>
    /// Heterogeneous dual-GPU cooperative execution across both NVIDIA dGPU and AMD APU.
    /// </summary>
    DualGpu = 3,

    /// <summary>
    /// Multi-threaded CPU execution using cache-blocked AVX-512 / AVX2 microkernels and dynamic core scaling.
    /// </summary>
    Cpu = 4,

    /// <summary>
    /// Targets 4th Generation Ada Lovelace Tensor Cores on NVIDIA RTX 4060 dGPU (WMMA hardware execution).
    /// Delivers 30–60+ TFLOPS throughput for half-precision and TensorFloat matrix multiplication.
    /// </summary>
    NvidiaTensorCore = 5
}
