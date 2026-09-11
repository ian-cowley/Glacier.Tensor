using System;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

public class GemmKernelTests
{
    [Theory]
    [InlineData(4, 4, 4)]
    [InlineData(8, 16, 8)]
    [InlineData(17, 23, 19)]
    [InlineData(64, 64, 64)]
    [InlineData(128, 64, 96)]
    public void Gemm_MatchesScalarGroundTruth(int M, int K, int N)
    {
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 42);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 84);
        using var c = new Tensor<float>(M, N);

        GemmKernels.MatMul(a, b, c);

        // Ground truth double precision multiply
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                double expected = 0.0;
                for (int k = 0; k < K; k++)
                {
                    expected += (double)a[m, k] * (double)b[k, n];
                }
                float actual = c[m, n];
                Assert.True(Math.Abs(actual - (float)expected) < 1e-4f, 
                    $"Mismatch at [{m},{n}]: expected {expected:F6}, got {actual:F6} for M={M}, K={K}, N={N}");
            }
        }
    }
}
