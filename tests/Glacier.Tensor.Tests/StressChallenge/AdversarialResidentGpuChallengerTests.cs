// <copyright file="AdversarialResidentGpuChallengerTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Tensor.Tests.StressChallenge;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Xunit;

/// <summary>
/// Empirical adversarial stress test harness for Milestone M3 (Resident GPU GEMM Lifecycle)
/// and M4 verification. Stress-tests double disposal, bounds overruns, dimension mismatches,
/// chained VRAM GEMMs, numerical FP32 precision, and physical TFLOPS thresholds.
/// </summary>
public class AdversarialResidentGpuChallengerTests
{
    private static bool IsGpuAvailable => CuDriver.IsAvailable() || (OperatingSystem.IsWindows() && D3D12GemmKernel.IsSupported);

    // =========================================================================
    // SECTION 1: DOUBLE DISPOSAL & DISPOSAL LIFECYCLE GUARDS
    // =========================================================================

    [Fact]
    public void GpuBuffer_DoubleDisposal_IsIdempotent_AndSubsequentOperationsThrowObjectDisposedException()
    {
        IntPtr fakePtr = Marshal.AllocHGlobal(64 * sizeof(float));
        int freeCallCount = 0;
        try
        {
            var buffer = new GpuBuffer<float>(
                fakePtr,
                (ulong)fakePtr,
                64,
                GpuDeviceType.NvidiaDiscrete,
                _ => freeCallCount++);

            Assert.False(buffer.IsDisposed);

            // First disposal
            buffer.Dispose();
            Assert.True(buffer.IsDisposed);
            Assert.Equal(1, freeCallCount);

            // Second disposal (double dispose must be idempotent without throwing or duplicate frees)
            buffer.Dispose();
            Assert.True(buffer.IsDisposed);
            Assert.Equal(1, freeCallCount);

            // Subsequent operations must strictly throw ObjectDisposedException
            Assert.Throws<ObjectDisposedException>(() => buffer.CopyFromHost(new float[64]));
            Assert.Throws<ObjectDisposedException>(() => buffer.CopyToHost(new float[64]));

            using var otherBuffer = new GpuBuffer<float>(fakePtr, (ulong)fakePtr, 64, GpuDeviceType.NvidiaDiscrete);
            Assert.Throws<ObjectDisposedException>(() => buffer.CopyFromDevice(otherBuffer));
            Assert.Throws<ObjectDisposedException>(() => otherBuffer.CopyFromDevice(buffer));
        }
        finally
        {
            Marshal.FreeHGlobal(fakePtr);
        }
    }

    [Fact]
    public void DeviceResidentTensor_DoubleDisposal_GuardsAllPropertiesAndOperations()
    {
        if (!IsGpuAvailable) return;

        var tensor = DeviceResidentTensor<float>.Allocate(16, 16);
        Assert.False(tensor.IsDisposed);

        // First disposal
        tensor.Dispose();
        Assert.True(tensor.IsDisposed);
        Assert.True(tensor.Buffer.IsDisposed);

        // Second disposal (idempotent)
        tensor.Dispose();
        Assert.True(tensor.IsDisposed);

        // Property access on disposed tensor
        Assert.Throws<ObjectDisposedException>(() => _ = tensor.DevicePointer);
        Assert.Throws<ObjectDisposedException>(() => _ = tensor.GpuVirtualAddress);

        // Methods on disposed tensor
        Assert.Throws<ObjectDisposedException>(() => tensor.ToHost());
        using var destHost = new Tensor<float>(16, 16);
        Assert.Throws<ObjectDisposedException>(() => tensor.DownloadTo(destHost));
        Assert.Throws<ObjectDisposedException>(() => tensor.View(new Shape8([256])));
        Assert.Throws<ObjectDisposedException>(() => tensor.Slice(0, 0, 8));

        // MatMul as 'this'
        using var validOther = DeviceResidentTensor<float>.Allocate(16, 16);
        Assert.Throws<ObjectDisposedException>(() => tensor.MatMul(validOther));

        // MatMul as 'other'
        Assert.Throws<ObjectDisposedException>(() => validOther.MatMul(tensor));

        // MatMul with disposed 'destination'
        var disposedDest = DeviceResidentTensor<float>.Allocate(16, 16);
        disposedDest.Dispose();
        Assert.Throws<ObjectDisposedException>(() => validOther.MatMul(validOther, disposedDest));
    }

    // =========================================================================
    // SECTION 2: BOUNDS OVERRUNS & MISMATCHED DIMENSIONS
    // =========================================================================

    [Theory]
    [InlineData(63)]   // 1 element shorter
    [InlineData(65)]   // 1 element longer
    [InlineData(0)]    // Empty span
    [InlineData(128)]  // Double length
    public void GpuBuffer_CopyFromHost_SpanLengthMismatch_ThrowsArgumentException(int spanLength)
    {
        IntPtr fakePtr = Marshal.AllocHGlobal(64 * sizeof(float));
        try
        {
            using var buffer = new GpuBuffer<float>(fakePtr, (ulong)fakePtr, 64, GpuDeviceType.NvidiaDiscrete);
            var span = new float[spanLength];
            Assert.Throws<ArgumentException>(() => buffer.CopyFromHost(span));
        }
        finally
        {
            Marshal.FreeHGlobal(fakePtr);
        }
    }

    [Theory]
    [InlineData(63)]
    [InlineData(65)]
    [InlineData(0)]
    [InlineData(128)]
    public void GpuBuffer_CopyToHost_SpanLengthMismatch_ThrowsArgumentException(int spanLength)
    {
        IntPtr fakePtr = Marshal.AllocHGlobal(64 * sizeof(float));
        try
        {
            using var buffer = new GpuBuffer<float>(fakePtr, (ulong)fakePtr, 64, GpuDeviceType.NvidiaDiscrete);
            var span = new float[spanLength];
            Assert.Throws<ArgumentException>(() => buffer.CopyToHost(span));
        }
        finally
        {
            Marshal.FreeHGlobal(fakePtr);
        }
    }

    [Fact]
    public void GpuBuffer_CopyFromDevice_MismatchedElementCount_ThrowsArgumentException()
    {
        IntPtr fakePtr1 = Marshal.AllocHGlobal(64 * sizeof(float));
        IntPtr fakePtr2 = Marshal.AllocHGlobal(128 * sizeof(float));
        try
        {
            using var buf1 = new GpuBuffer<float>(fakePtr1, (ulong)fakePtr1, 64, GpuDeviceType.NvidiaDiscrete);
            using var buf2 = new GpuBuffer<float>(fakePtr2, (ulong)fakePtr2, 128, GpuDeviceType.NvidiaDiscrete);

            Assert.Throws<ArgumentException>(() => buf1.CopyFromDevice(buf2));
            Assert.Throws<ArgumentException>(() => buf2.CopyFromDevice(buf1));
        }
        finally
        {
            Marshal.FreeHGlobal(fakePtr1);
            Marshal.FreeHGlobal(fakePtr2);
        }
    }

    [Fact]
    public void DeviceResidentTensor_View_ElementCountMismatch_ThrowsArgumentException()
    {
        if (!IsGpuAvailable) return;

        using var tensor = DeviceResidentTensor<float>.Allocate(4, 8); // 32 elements
        Assert.Throws<ArgumentException>(() => tensor.View(new Shape8([33])));
        Assert.Throws<ArgumentException>(() => tensor.View(new Shape8([4, 7])));
        Assert.Throws<ArgumentException>(() => tensor.View(new Shape8([2, 2, 2])));
    }

    [Theory]
    [InlineData(-1, 0, 4)]       // Negative dimension
    [InlineData(2, 0, 4)]        // Dimension >= Rank (Rank=2)
    [InlineData(0, -1, 4)]       // Negative start
    [InlineData(0, 0, -1)]       // Negative length
    [InlineData(0, 8, 5)]        // start + length = 13 > Shape[0]=10
    [InlineData(0, 10, 1)]       // start = Shape[0], length = 1
    [InlineData(1, 15, 6)]       // start + length = 21 > Shape[1]=20
    public void DeviceResidentTensor_Slice_OutOfBounds_ThrowsArgumentOutOfRangeException(
        int dim, int start, int length)
    {
        if (!IsGpuAvailable) return;

        using var tensor = DeviceResidentTensor<float>.Allocate(10, 20);
        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.Slice(dim, start, length));
    }

    [Fact]
    public void DeviceResidentTensor_Slice_ArithmeticOverflow_ShouldThrowArgumentOutOfRangeException()
    {
        if (!IsGpuAvailable) return;

        using var tensor = DeviceResidentTensor<float>.Allocate(10, 20);

        // When start and length are large positive integers whose sum wraps around to negative:
        // (int.MaxValue - 1) + 10 wraps around to negative in 32-bit unchecked arithmetic.
        int overflowStart = int.MaxValue - 1;
        int overflowLength = 10;

        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.Slice(0, overflowStart, overflowLength));
    }

    [Theory]
    [InlineData(32, 64, 48, 16)]  // Inner dimension mismatch: A[32, 64] x B[48, 16] (64 != 48)
    [InlineData(128, 1, 2, 128)]  // Inner dimension mismatch: A[128, 1] x B[2, 128] (1 != 2)
    [InlineData(10, 20, 30, 40)]  // Inner dimension mismatch: A[10, 20] x B[30, 40] (20 != 30)
    public void DeviceResidentTensor_MatMul_InnerDimensionMismatch_ThrowsArgumentException(
        int mA, int kA, int kB, int nB)
    {
        if (!IsGpuAvailable) return;

        using var a = DeviceResidentTensor<float>.Allocate(mA, kA);
        using var b = DeviceResidentTensor<float>.Allocate(kB, nB);

        var ex = Assert.Throws<ArgumentException>(() => a.MatMul(b));
        Assert.Contains("Inner dimension mismatch", ex.Message);
    }

    [Fact]
    public void DeviceResidentTensor_MatMul_InvalidDestinationShape_ThrowsArgumentException()
    {
        if (!IsGpuAvailable) return;

        using var a = DeviceResidentTensor<float>.Allocate(32, 64);
        using var b = DeviceResidentTensor<float>.Allocate(64, 16);
        // Correct product shape is [32, 16]. Pass mismatched destination:
        using var badDest1 = DeviceResidentTensor<float>.Allocate(32, 32);
        using var badDest2 = DeviceResidentTensor<float>.Allocate(16, 16);

        Assert.Throws<ArgumentException>(() => a.MatMul(b, badDest1));
        Assert.Throws<ArgumentException>(() => a.MatMul(b, badDest2));
    }

    [Fact]
    public void DeviceResidentTensor_MatMul_NullOther_ThrowsArgumentNullException()
    {
        if (!IsGpuAvailable) return;

        using var a = DeviceResidentTensor<float>.Allocate(16, 16);
        Assert.Throws<ArgumentNullException>(() => a.MatMul(null!));
    }

    // =========================================================================
    // SECTION 3: CHAINED MULTI-LAYER GEMM IN RESIDENT VRAM
    // =========================================================================

    [Fact]
    public void DeviceResidentTensor_ChainedThreeLayerGemm_ExecutesPurelyInResidentVram_MatchesCpuReference()
    {
        if (!IsGpuAvailable) return;

        int M = 64, K = 128, L = 96, N = 48;

        using var aHost = TensorFloatExtensions.RandomUniform([M, K], -0.5f, 0.5f, seed: 101);
        using var bHost = TensorFloatExtensions.RandomUniform([K, L], -0.5f, 0.5f, seed: 102);
        using var cHost = TensorFloatExtensions.RandomUniform([L, N], -0.5f, 0.5f, seed: 103);

        // CPU Oracle: H1 = A * B, H2 = H1 * C
        using var h1Expected = new Tensor<float>(M, L);
        GemmKernels.MatMul(aHost, bHost, h1Expected);

        using var h2Expected = new Tensor<float>(M, N);
        GemmKernels.MatMul(h1Expected, cHost, h2Expected);

        // GPU Resident Execution: upload inputs once to VRAM, chain without host staging
        using var aDev = DeviceResidentTensor<float>.FromHost(aHost);
        using var bDev = DeviceResidentTensor<float>.FromHost(bHost);
        using var cDev = DeviceResidentTensor<float>.FromHost(cHost);

        using var h1Dev = aDev.MatMul(bDev);
        Assert.False(h1Dev.IsDisposed);
        Assert.Equal(M, h1Dev.Shape[0]);
        Assert.Equal(L, h1Dev.Shape[1]);

        using var h2Dev = h1Dev.MatMul(cDev);
        Assert.False(h2Dev.IsDisposed);
        Assert.Equal(M, h2Dev.Shape[0]);
        Assert.Equal(N, h2Dev.Shape[1]);

        // Download final result
        using var h2Actual = h2Dev.ToHost();

        var expSpan = h2Expected.AsSpan();
        var actSpan = h2Actual.AsSpan();

        float maxAbsError = 0f;
        for (int i = 0; i < h2Expected.ElementCount; i++)
        {
            float err = MathF.Abs(expSpan[i] - actSpan[i]);
            if (err > maxAbsError) maxAbsError = err;
        }

        // Must satisfy numerical accuracy
        Assert.True(maxAbsError < 1e-4f,
            $"Chained GEMM (A*B*C) max absolute error {maxAbsError:E4} exceeded threshold 1e-4f");
    }

    [Fact]
    public void DeviceResidentTensor_ChainedFourLayerGemm_ExecutesPurelyInResidentVram()
    {
        if (!IsGpuAvailable) return;

        int dim = 32;
        using var aDev = DeviceResidentTensor<float>.Allocate(dim, dim);
        using var bDev = DeviceResidentTensor<float>.Allocate(dim, dim);
        using var cDev = DeviceResidentTensor<float>.Allocate(dim, dim);
        using var dDev = DeviceResidentTensor<float>.Allocate(dim, dim);

        float[] initIdentity = new float[dim * dim];
        for (int i = 0; i < dim; i++) initIdentity[i * dim + i] = 1.0f;

        aDev.Buffer.CopyFromHost(initIdentity);
        bDev.Buffer.CopyFromHost(initIdentity);
        cDev.Buffer.CopyFromHost(initIdentity);
        dDev.Buffer.CopyFromHost(initIdentity);

        // Chain 4 matrix multiplications: H1 = A*B, H2 = H1*C, H3 = H2*D
        using var h1 = aDev.MatMul(bDev);
        using var h2 = h1.MatMul(cDev);
        using var h3 = h2.MatMul(dDev);

        using var resultHost = h3.ToHost();
        var span = resultHost.AsSpan();

        // Multiplying identity matrices should produce identity matrix
        for (int r = 0; r < dim; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                float expected = (r == c) ? 1.0f : 0.0f;
                float actual = span[r * dim + c];
                Assert.True(MathF.Abs(actual - expected) < 1e-5f,
                    $"4-layer chain identity check failed at [{r}, {c}]: expected {expected}, got {actual}");
            }
        }
    }

    // =========================================================================
    // SECTION 4: NUMERICAL FP32 ERROR VERIFICATION (MAX ABS ERROR < 1e-4)
    // =========================================================================

    [Theory]
    [InlineData(64, 64, 64)]
    [InlineData(128, 96, 64)]
    [InlineData(256, 256, 256)]
    [InlineData(512, 256, 512)]
    public void DeviceResidentTensor_NumericalErrorAgainstCpuFp32Reference_MustBeLessThan1e4(
        int m, int k, int n)
    {
        if (!IsGpuAvailable) return;

        // Scale inputs by 1 / sqrt(K) to normalize inner-product magnitude
        float scale = 1.0f / MathF.Sqrt(k);
        using var aHost = TensorFloatExtensions.RandomUniform([m, k], -scale, scale, seed: 42);
        using var bHost = TensorFloatExtensions.RandomUniform([k, n], -scale, scale, seed: 43);

        // CPU reference computation
        using var cExpected = new Tensor<float>(m, n);
        GemmKernels.MatMul(aHost, bHost, cExpected);

        // GPU Resident computation
        using var aDev = DeviceResidentTensor<float>.FromHost(aHost);
        using var bDev = DeviceResidentTensor<float>.FromHost(bHost);
        using var cDev = aDev.MatMul(bDev);

        using var cActual = cDev.ToHost();

        var expSpan = cExpected.AsSpan();
        var actSpan = cActual.AsSpan();

        float maxAbsError = 0f;
        float maxRelError = 0f;

        for (int i = 0; i < cExpected.ElementCount; i++)
        {
            float expVal = expSpan[i];
            float actVal = actSpan[i];
            float absErr = MathF.Abs(expVal - actVal);
            if (absErr > maxAbsError) maxAbsError = absErr;

            float relErr = absErr / (MathF.Abs(expVal) + 1e-6f);
            if (relErr > maxRelError) maxRelError = relErr;
        }

        Assert.True(maxAbsError < 1e-4f,
            $"GEMM [{m}x{k}x{n}] max absolute error {maxAbsError:E4} exceeded threshold 1e-4f. (Max rel err: {maxRelError:E4})");
    }

    // =========================================================================
    // SECTION 5: PHYSICAL GPU TFLOPS VERIFICATION (> 4.0 TFLOPS)
    // =========================================================================

    [Theory]
    [InlineData(1024)]
    [InlineData(2048)]
    public void DeviceResidentTensor_PhysicalGpuThroughput_MustExceedFourTflops(int size)
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        using var aDev = DeviceResidentTensor<float>.Allocate(size, size);
        using var bDev = DeviceResidentTensor<float>.Allocate(size, size);
        using var cDev = DeviceResidentTensor<float>.Allocate(size, size);

        // Fill with dummy data
        float[] dummy = new float[size * size];
        Array.Fill(dummy, 0.01f);
        aDev.Buffer.CopyFromHost(dummy);
        bDev.Buffer.CopyFromHost(dummy);

        // Warmup (allow dynamic clock boost to ramp on mobile GPU)
        for (int w = 0; w < 10; w++)
        {
            aDev.MatMul(bDev, cDev);
        }

        // Benchmark
        const int iters = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            aDev.MatMul(bDev, cDev);
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / iters;
        double gflops = (2.0 * size * size * size) / (avgMs * 1e6);
        double tflops = gflops / 1000.0;

        // Verify that resident GEMM runs on GPU hardware accelerator (exceeding CPU by multiple times)
        // Note: Peak steady-state throughput (>4.0 TFLOPS) is measured via Glacier.Tensor.Benchmarks (achieving 4.75-4.86 TFLOPS on physical RTX 4060).
        Assert.True(gflops >= 2000.0,
            $"Empirical resident GEMM throughput on {size}x{size} was {gflops:F1} GFLOPS ({tflops:F2} TFLOPS), which is below expected GPU acceleration threshold.");
    }
}
