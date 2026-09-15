using System;
using BenchmarkDotNet.Attributes;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Benchmarks;

[MemoryDiagnoser]
public class GemmBenchmarks : IDisposable
{
    private Tensor<float>? _matA;
    private Tensor<float>? _matB;
    private Tensor<float>? _matC;

    [Params(256, 512)]
    public int Size { get; set; } = 512;

    [GlobalSetup]
    public void Setup()
    {
        _matA = TensorFloatExtensions.RandomUniform([Size, Size], -1f, 1f, seed: 42);
        _matB = TensorFloatExtensions.RandomUniform([Size, Size], -1f, 1f, seed: 84);
        _matC = new Tensor<float>(Size, Size);
    }

    [Benchmark(Description = "Glacier.Tensor Cache-Blocked SIMD GEMM")]
    public void BenchmarkGemm()
    {
        GemmKernels.MatMul(_matA!, _matB!, _matC!);
    }

    [Benchmark(Description = "Glacier.Tensor Direct3D 12 GPU GEMM")]
    public void BenchmarkDirect3D12()
    {
        if (GpuAccelerator.HasDirect3D12)
        {
            _matA!.MatMul(_matB!, _matC!, GpuTarget.Direct3D12);
        }
    }

    [Benchmark(Description = "Glacier.Tensor Vulkan GPU GEMM")]
    public void BenchmarkVulkan()
    {
        if (GpuAccelerator.HasVulkan)
        {
            _matA!.MatMul(_matB!, _matC!, GpuTarget.Vulkan);
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _matA?.Dispose();
        _matB?.Dispose();
        _matC?.Dispose();
    }

    public void Dispose() => Cleanup();
}
