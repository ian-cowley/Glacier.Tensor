using System;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

public class GpuAcceleratorTests
{
    [Fact]
    public void GpuAccelerator_ReportsHardwareCapabilities()
    {
        // Must never throw and report boolean capabilities cleanly
        bool hasNvidia = GpuAccelerator.HasNvidiaGpu;
        bool hasAmd = GpuAccelerator.HasAmdGpu;
        bool isGpuAvail = GpuAccelerator.IsGpuAvailable;

        Assert.Equal(hasNvidia || hasAmd, isGpuAvail);
    }

    [Theory]
    [InlineData(32, 32, 32)]
    [InlineData(64, 48, 80)]
    [InlineData(128, 128, 128)]
    [InlineData(256, 128, 192)]
    public void MatMul_Auto_MatchesGroundTruth(int M, int K, int N)
    {
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 101);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 202);
        using var c = new Tensor<float>(M, N);

        a.MatMul(b, c, GpuTarget.Auto);

        VerifyAgainstGroundTruth(a, b, c, M, K, N);
    }

    [Fact]
    public void MatMul_CpuTarget_MatchesGroundTruth()
    {
        int M = 64, K = 64, N = 64;
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 303);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 404);
        using var c = a.MatMul(b, GpuTarget.Cpu);

        VerifyAgainstGroundTruth(a, b, c, M, K, N);
    }

    [Fact]
    public void MatMul_Nvidia_IfHardwareAvailable_ComputesAccurateResult()
    {
        if (!GpuAccelerator.HasNvidiaGpu)
        {
            // Skip test if no NVIDIA GPU driver present
            return;
        }

        int M = 256, K = 256, N = 256;
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 505);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 606);
        using var c = new Tensor<float>(M, N);

        a.MatMul(b, c, GpuTarget.Nvidia);

        VerifyAgainstGroundTruth(a, b, c, M, K, N, tolerance: 1e-3f);
    }

    [Fact]
    public void MatMul_Amd_IfHardwareAvailable_ComputesAccurateResult()
    {
        if (!GpuAccelerator.HasAmdGpu)
        {
            // Skip test if no AMD GPU driver present
            return;
        }

        int M = 128, K = 128, N = 128;
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 707);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 808);
        using var c = new Tensor<float>(M, N);

        a.MatMul(b, c, GpuTarget.Amd);

        VerifyAgainstGroundTruth(a, b, c, M, K, N, tolerance: 1e-3f);
    }

    [Fact]
    public void MatMul_DualGpu_ExecutesOrFallsBackGracefully()
    {
        int M = 128, K = 64, N = 96;
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 909);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 1010);
        using var c = new Tensor<float>(M, N);

        a.MatMul(b, c, GpuTarget.DualGpu);

        VerifyAgainstGroundTruth(a, b, c, M, K, N, tolerance: 1e-3f);
    }

    [Fact]
    public void MatMul_NvidiaTensorCore_IfHardwareAvailable_ComputesAccurateResult()
    {
        if (!GpuAccelerator.HasNvidiaGpu)
        {
            return;
        }

        int M = 256, K = 256, N = 256;
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 1111);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 1212);
        using var c = new Tensor<float>(M, N);

        a.MatMul(b, c, GpuTarget.NvidiaTensorCore);

        // Half precision Tensor Core inputs have ~1e-2 tolerance due to 11-bit mantissa conversion
        VerifyAgainstGroundTruth(a, b, c, M, K, N, tolerance: 0.05f);
    }

    private static void VerifyAgainstGroundTruth(Tensor<float> a, Tensor<float> b, Tensor<float> c, int M, int K, int N, float tolerance = 1e-4f)
    {
        var spanA = a.AsSpan();
        var spanB = b.AsSpan();
        var spanC = c.AsSpan();

        // Sample checkpoints to verify accuracy without slow O(M*K*N) CPU overhead on big tests
        int stepM = Math.Max(1, M / 16);
        int stepN = Math.Max(1, N / 16);

        for (int m = 0; m < M; m += stepM)
        {
            for (int n = 0; n < N; n += stepN)
            {
                double expected = 0.0;
                for (int k = 0; k < K; k++)
                {
                    expected += (double)spanA[m * K + k] * (double)spanB[k * N + n];
                }

                float actual = spanC[m * N + n];
                Assert.True(Math.Abs(actual - (float)expected) <= tolerance,
                    $"Mismatch at [{m},{n}]: expected {expected:F6}, got {actual:F6} (diff={Math.Abs(actual - (float)expected):E2})");
            }
        }
    }
}
