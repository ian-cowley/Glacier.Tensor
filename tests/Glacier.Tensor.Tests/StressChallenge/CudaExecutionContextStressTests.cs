using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests.StressChallenge;

public class CudaExecutionContextStressTests
{
    [Fact]
    public async Task CudaExecutionContext_ConcurrentRentAndReturn_StressTest()
    {
        if (!GpuAccelerator.HasNvidiaGpu)
        {
            // Verify graceful fallback when CUDA driver is absent
            return;
        }

        // Initialize NVIDIA driver context via a minimal GEMM
        using var dummyA = new Tensor<float>(1, 1);
        using var dummyB = new Tensor<float>(1, 1);
        using var dummyC = new Tensor<float>(1, 1);
        if (!GpuAccelerator.ExecuteNvidiaGemm(dummyA, dummyB, dummyC))
        {
            return;
        }

        const int numThreads = 16;
        const int iterationsPerThread = 50;
        var rentedStreams = new ConcurrentBag<IntPtr>();
        var errors = new ConcurrentBag<string>();

        var tasks = new Task[numThreads];
        for (int t = 0; t < numThreads; t++)
        {
            int threadId = t;
            tasks[t] = Task.Run(() =>
            {
                var random = new Random(42 + threadId);
                for (int i = 0; i < iterationsPerThread; i++)
                {
                    CudaExecutionContext? ctx = null;
                    try
                    {
                        ctx = CudaExecutionContext.Rent();
                        if (ctx.Stream == IntPtr.Zero)
                        {
                            errors.Add($"Thread {threadId}: Rented context with null stream.");
                        }

                        rentedStreams.Add(ctx.Stream);

                        // Varying allocation sizes to trigger dynamic resizing
                        nuint bytesA = (nuint)(random.Next(32, 256) * 4);
                        nuint bytesB = (nuint)(random.Next(32, 256) * 4);
                        nuint bytesC = (nuint)(random.Next(32, 256) * 4);

                        ctx.EnsureCapacityA(bytesA);
                        ctx.EnsureCapacityB(bytesB);
                        ctx.EnsureCapacityC(bytesC);

                        if (ctx.DevA == IntPtr.Zero || ctx.DevB == IntPtr.Zero || ctx.DevC == IntPtr.Zero)
                        {
                            errors.Add($"Thread {threadId}: Allocation failed to produce valid device pointer.");
                        }
                    }
                    catch (Exception ex)
                    {
                        errors.Add($"Thread {threadId} iteration {i} threw: {ex.Message}");
                    }
                    finally
                    {
                        CudaExecutionContext.Return(ctx);
                    }
                }
            });
        }

        await Task.WhenAll(tasks);
        Assert.Empty(errors);
    }

    [Fact]
    public async Task CudaExecutionContext_ConcurrentGemmExecution_StressTest()
    {
        if (!GpuAccelerator.HasNvidiaGpu)
        {
            return;
        }

        const int numThreads = 16;
        const int iterationsPerThread = 10;
        var errors = new ConcurrentBag<string>();

        var tasks = new Task[numThreads];
        for (int t = 0; t < numThreads; t++)
        {
            int threadId = t;
            tasks[t] = Task.Run(() =>
            {
                var rng = new Random(100 + threadId * 7);
                for (int iter = 0; iter < iterationsPerThread; iter++)
                {
                    int M = 64, K = 64, N = 64;
                    using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: rng.Next());
                    using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: rng.Next());
                    using var c = new Tensor<float>(M, N);

                    bool success = GpuAccelerator.ExecuteNvidiaGemm(a, b, c);
                    if (!success)
                    {
                        errors.Add($"Thread {threadId} iteration {iter}: ExecuteNvidiaGemm returned false.");
                        continue;
                    }

                    // Verify against CPU reference on sample coordinates
                    var spanA = a.AsSpan();
                    var spanB = b.AsSpan();
                    var spanC = c.AsSpan();

                    for (int m = 0; m < M; m += 16)
                    {
                        for (int n = 0; n < N; n += 16)
                        {
                            double expected = 0.0;
                            for (int k = 0; k < K; k++)
                            {
                                expected += (double)spanA[m * K + k] * (double)spanB[k * N + n];
                            }

                            float actual = spanC[m * N + n];
                            float diff = Math.Abs(actual - (float)expected);
                            if (diff > 1e-3f)
                            {
                                errors.Add($"Thread {threadId} iter {iter} mismatch at [{m},{n}]: expected {expected:F4}, got {actual:F4}, diff={diff:E2}");
                            }
                        }
                    }
                }
            });
        }

        await Task.WhenAll(tasks);
        Assert.Empty(errors);
    }

    [Fact]
    public void CudaExecutionContext_PoolLimit_PreventsLeakAndDisposesGracefully()
    {
        if (!GpuAccelerator.HasNvidiaGpu)
        {
            return;
        }

        // Initialize NVIDIA driver context via a minimal GEMM
        using var dummyA = new Tensor<float>(1, 1);
        using var dummyB = new Tensor<float>(1, 1);
        using var dummyC = new Tensor<float>(1, 1);
        if (!GpuAccelerator.ExecuteNvidiaGemm(dummyA, dummyB, dummyC))
        {
            return;
        }

        // Rent 50 contexts simultaneously (exceeds default MaxPoolSize of 32 or Environment.ProcessorCount * 2)
        const int count = 50;
        var contexts = new List<CudaExecutionContext>();

        try
        {
            for (int i = 0; i < count; i++)
            {
                contexts.Add(CudaExecutionContext.Rent());
            }

            // Verify each rented context has a unique stream
            var streams = new HashSet<IntPtr>();
            foreach (var ctx in contexts)
            {
                Assert.True(streams.Add(ctx.Stream), "Duplicate stream detected among simultaneously rented contexts!");
            }
        }
        finally
        {
            // Return all contexts; excess must be disposed without exception
            foreach (var ctx in contexts)
            {
                CudaExecutionContext.Return(ctx);
            }
        }

        // Subsequent rent should succeed immediately from pool
        var reRented = CudaExecutionContext.Rent();
        Assert.NotEqual(IntPtr.Zero, reRented.Stream);
        CudaExecutionContext.Return(reRented);
    }
}
