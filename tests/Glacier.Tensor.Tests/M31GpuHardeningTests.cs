using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Glacier.Gpu.Drivers;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

[Trait("Hardware", "Gpu")]
public unsafe class M31GpuHardeningTests
{
    [Theory]
    [InlineData(1, 7, 13)]
    [InlineData(17, 19, 23)]
    [InlineData(65, 33, 67)]
    public void TensorCoreGemm_Fp32_Non16Aligned_ZeroOobCorruption_AndMatchesCpu(int m, int k, int n)
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        using var dummyA = new Tensor<float>(1, 1);
        using var dummyB = new Tensor<float>(1, 1);
        using var dummyC = new Tensor<float>(1, 1);
        if (!GpuAccelerator.ExecuteNvidiaGemm(dummyA, dummyB, dummyC)) return;

        using var a = TensorFloatExtensions.RandomUniform([m, k], -1f, 1f, seed: 100 + m);
        using var b = TensorFloatExtensions.RandomUniform([k, n], -1f, 1f, seed: 200 + n);
        using var cCpu = new Tensor<float>(m, n);
        GemmKernels.MatMul(a, b, cCpu);

        const int padElems = 64;
        const float canary = -12345.678f;
        int totalC = m * n + 2 * padElems;
        float[] hostBuf = new float[totalC];
        Array.Fill(hostBuf, canary);

        Assert.True(GpuAccelerator.HasNvidiaGpu);
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr devA, (nuint)(m * k * sizeof(float))), "MemAlloc A");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr devB, (nuint)(k * n * sizeof(float))), "MemAlloc B");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr devCFull, (nuint)(totalC * sizeof(float))), "MemAlloc C");
        try
        {
            fixed (float* pA = a.AsSpan(), pB = b.AsSpan(), pHost = hostBuf)
            {
                CuDriver.Check(CuDriver.MemcpyHtoD(devA, (IntPtr)pA, (nuint)(m * k * sizeof(float))), "HtoD A");
                CuDriver.Check(CuDriver.MemcpyHtoD(devB, (IntPtr)pB, (nuint)(k * n * sizeof(float))), "HtoD B");
                CuDriver.Check(CuDriver.MemcpyHtoD(devCFull, (IntPtr)pHost, (nuint)(totalC * sizeof(float))), "HtoD C");

                IntPtr devCMid = devCFull + (nint)(padElems * sizeof(float));
                bool ok = GpuAccelerator.ExecuteNvidiaTensorCoreGemm(devA, devB, devCMid, m, n, k);
                Assert.True(ok);

                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pHost, devCFull, (nuint)(totalC * sizeof(float))), "DtoH C");
            }

            for (int i = 0; i < padElems; i++)
            {
                Assert.Equal(canary, hostBuf[i]);
                Assert.Equal(canary, hostBuf[padElems + m * n + i]);
            }

            var exp = cCpu.AsSpan();
            for (int i = 0; i < m * n; i++)
            {
                Assert.True(MathF.Abs(hostBuf[padElems + i] - exp[i]) <= 0.1f,
                    $"FP32 TensorCore mismatch at [{i}] ({m}x{k}x{n}): got {hostBuf[padElems + i]}, expected {exp[i]}");
            }
        }
        finally
        {
            CuDriver.MemFree(devA);
            CuDriver.MemFree(devB);
            CuDriver.MemFree(devCFull);
        }
    }

    [Theory]
    [InlineData(1, 7, 13)]
    [InlineData(17, 19, 23)]
    [InlineData(65, 33, 67)]
    public void TensorCoreGemm_Fp16_Non16Aligned_ComputesAccurateResult(int m, int k, int n)
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        using var aFp32 = TensorFloatExtensions.RandomUniform([m, k], -1f, 1f, seed: 300 + m);
        using var bFp32 = TensorFloatExtensions.RandomUniform([k, n], -1f, 1f, seed: 400 + n);
        using var aHalf = new Tensor<Half>(m, k);
        using var bHalf = new Tensor<Half>(k, n);
        using var cActual = new Tensor<float>(m, n);
        using var cCpu = new Tensor<float>(m, n);

        var aSrc = aFp32.AsSpan();
        var bSrc = bFp32.AsSpan();
        var aDst = aHalf.AsSpan();
        var bDst = bHalf.AsSpan();
        for (int i = 0; i < aSrc.Length; i++) aDst[i] = (Half)aSrc[i];
        for (int i = 0; i < bSrc.Length; i++) bDst[i] = (Half)bSrc[i];

        bool ok = GpuAccelerator.ExecuteNvidiaTensorCoreGemm(aHalf, bHalf, cActual);
        Assert.True(ok);

        GemmKernels.MatMul(aFp32, bFp32, cCpu);
        var act = cActual.AsSpan();
        var exp = cCpu.AsSpan();
        for (int i = 0; i < exp.Length; i++)
        {
            Assert.True(MathF.Abs(act[i] - exp[i]) <= 0.15f,
                $"FP16 TensorCore mismatch at [{i}] ({m}x{k}x{n}): got {act[i]}, expected {exp[i]}");
        }
    }

    [Theory]
    [InlineData(1, 7, 13)]
    [InlineData(17, 19, 23)]
    [InlineData(65, 33, 67)]
    [InlineData(129, 65, 131)]
    public void D3D12GemmKernel_Non16Aligned_MatchesCpuReference(int m, int k, int n)
    {
        if (!D3D12GemmKernel.IsSupported) return;

        using var a = TensorFloatExtensions.RandomUniform([m, k], -1f, 1f, seed: 500 + m);
        using var b = TensorFloatExtensions.RandomUniform([k, n], -1f, 1f, seed: 600 + n);
        using var cGpu = new Tensor<float>(m, n);
        using var cCpu = new Tensor<float>(m, n);

        bool ok = D3D12GemmKernel.Execute(a, b, cGpu, out var diag);
        Assert.True(ok);
        Assert.NotNull(diag);
        Assert.True(diag.Success);

        GemmKernels.MatMul(a, b, cCpu);
        var g = cGpu.AsSpan();
        var c = cCpu.AsSpan();
        for (int i = 0; i < c.Length; i++)
        {
            Assert.True(MathF.Abs(g[i] - c[i]) <= 1e-3f,
                $"D3D12 mismatch at [{i}] ({m}x{k}x{n}): GPU={g[i]}, CPU={c[i]}");
        }
    }

    [Theory]
    [InlineData(1, 7, 13)]
    [InlineData(17, 19, 23)]
    [InlineData(65, 33, 67)]
    [InlineData(129, 65, 131)]
    public void VulkanGemmKernel_Non16Aligned_MatchesCpuReference(int m, int k, int n)
    {
        if (!VulkanGemmKernel.IsSupported) return;

        using var a = TensorFloatExtensions.RandomUniform([m, k], -1f, 1f, seed: 700 + m);
        using var b = TensorFloatExtensions.RandomUniform([k, n], -1f, 1f, seed: 800 + n);
        using var cGpu = new Tensor<float>(m, n);
        using var cCpu = new Tensor<float>(m, n);

        bool ok = VulkanGemmKernel.Execute(a, b, cGpu);
        Assert.True(ok);

        GemmKernels.MatMul(a, b, cCpu);
        var g = cGpu.AsSpan();
        var c = cCpu.AsSpan();
        for (int i = 0; i < c.Length; i++)
        {
            Assert.True(MathF.Abs(g[i] - c[i]) <= 1e-3f,
                $"Vulkan mismatch at [{i}] ({m}x{k}x{n}): GPU={g[i]}, CPU={c[i]}");
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void GpuAccelerator_AcceleratedMatMul_AllTransposeModes_MatchCpuReference(bool transposeA, bool transposeB)
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        int m = 135, k = 129, n = 141;
        using var a = TensorFloatExtensions.RandomUniform(transposeA ? [k, m] : [m, k], -1f, 1f, seed: 91);
        using var b = TensorFloatExtensions.RandomUniform(transposeB ? [n, k] : [k, n], -1f, 1f, seed: 92);
        using var cGpu = new Tensor<float>(m, n);
        using var cCpu = new Tensor<float>(m, n);

        GpuAccelerator.AcceleratedMatMul(a, b, cGpu, transposeA, transposeB, GpuTarget.Nvidia);
        GpuAccelerator.AcceleratedMatMul(a, b, cCpu, transposeA, transposeB, GpuTarget.Cpu);

        var g = cGpu.AsSpan();
        var c = cCpu.AsSpan();
        for (int i = 0; i < c.Length; i++)
        {
            Assert.True(MathF.Abs(g[i] - c[i]) <= 2e-3f,
                $"Transposed GPU MatMul ({transposeA},{transposeB}) mismatch at [{i}]: GPU={g[i]}, CPU={c[i]}");
        }
    }

    [Fact]
    public void GpuAccelerator_NonContiguousInputs_SucceedWithoutSilentSwallowing()
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        using var aBase = TensorFloatExtensions.RandomUniform([140, 130], -1f, 1f, seed: 71);
        using var bBase = TensorFloatExtensions.RandomUniform([130, 140], -1f, 1f, seed: 72);
        using var aT = aBase.Transpose(); // non-contiguous [130, 140]
        using var bT = bBase.Transpose(); // non-contiguous [140, 130]
        using var cGpu = new Tensor<float>(130, 130);
        using var cCpu = new Tensor<float>(130, 130);

        Assert.True(GpuAccelerator.ExecuteNvidiaGemm(aT, bT, cGpu));
        GemmKernels.MatMul(aT, bT, cCpu);

        var g = cGpu.AsSpan();
        var c = cCpu.AsSpan();
        for (int i = 0; i < c.Length; i++)
        {
            Assert.True(MathF.Abs(g[i] - c[i]) <= 2e-3f);
        }
    }

    [Fact]
    public void CudaExecutionContext_DoubleReturnIsIdempotent_AndOversizedBuffersShrink()
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        using var dummyA = new Tensor<float>(1, 1);
        using var dummyB = new Tensor<float>(1, 1);
        using var dummyC = new Tensor<float>(1, 1);
        if (!GpuAccelerator.ExecuteNvidiaGemm(dummyA, dummyB, dummyC)) return;

        CudaExecutionContext.ClearPool();
        Assert.Equal(0, CudaExecutionContext.PooledCount);

        var ctx = CudaExecutionContext.Rent();
        ctx.EnsureCapacityA(80u * 1024u * 1024u);
        Assert.True(ctx.CapA >= 80u * 1024u * 1024u);

        CudaExecutionContext.Return(ctx);
        int countAfterFirst = CudaExecutionContext.PooledCount;
        CudaExecutionContext.Return(ctx); // Double return must be ignored cleanly
        int countAfterSecond = CudaExecutionContext.PooledCount;

        Assert.Equal(1, countAfterFirst);
        Assert.Equal(1, countAfterSecond);
        Assert.True(ctx.CapA <= 64u * 1024u * 1024u);

        CudaExecutionContext.TrimPool(0);
        Assert.Equal(0, CudaExecutionContext.PooledCount);
    }

    [Fact]
    public void ExecuteAmdGemm_ExecutesStrictlyOnAmdVendor_AndMatchesCpu()
    {
        const int m = 65, k = 33, n = 67;
        using var a = TensorFloatExtensions.RandomUniform([m, k], -1f, 1f, seed: 1101);
        using var b = TensorFloatExtensions.RandomUniform([k, n], -1f, 1f, seed: 1102);
        using var cCpu = new Tensor<float>(m, n);
        GemmKernels.MatMul(a, b, cCpu);

        if (!GpuAccelerator.HasAmdGpu)
        {
            using var cNoAmd = new Tensor<float>(m, n);
            Assert.False(GpuAccelerator.ExecuteAmdGemm(a, b, cNoAmd));
            Assert.False(D3D12GemmKernel.IsAmdSupported);
            Assert.False(VulkanGemmKernel.IsAmdSupported);
            Assert.False(D3D12GemmKernel.ExecuteAmd(a, b, cNoAmd));
            Assert.False(VulkanGemmKernel.ExecuteAmd(a, b, cNoAmd));
            Assert.False(GpuAccelerator.ExecuteDualGpuGemm(a, b, cNoAmd));
            return;
        }

        Assert.True(D3D12GemmKernel.IsAmdSupported || VulkanGemmKernel.IsAmdSupported,
            "HasAmdGpu is true, so at least one AMD compute backend (D3D12/Vulkan) must initialize.");

        using var cAmd = new Tensor<float>(m, n);
        Assert.True(GpuAccelerator.ExecuteAmdGemm(a, b, cAmd));
        AssertMatrixClose(cAmd, cCpu, 1e-3f, "ExecuteAmdGemm");

        if (OperatingSystem.IsWindows() && D3D12GemmKernel.IsAmdSupported)
        {
            Assert.Equal(0x1002u, D3D12GemmKernel.AmdVendorId);
            Assert.DoesNotContain("NVIDIA", D3D12GemmKernel.AmdDeviceName, StringComparison.OrdinalIgnoreCase);

            using var cD3d = new Tensor<float>(m, n);
            Assert.True(D3D12GemmKernel.ExecuteAmd(a, b, cD3d, out var diag));
            Assert.NotNull(diag);
            Assert.True(diag.Success);
            AssertMatrixClose(cD3d, cCpu, 1e-3f, "D3D12GemmKernel.ExecuteAmd");
        }

        if (VulkanGemmKernel.IsAmdSupported)
        {
            Assert.Equal(0x1002u, VulkanGemmKernel.AmdVendorId);
            Assert.DoesNotContain("NVIDIA", VulkanGemmKernel.AmdDeviceName, StringComparison.OrdinalIgnoreCase);

            using var cVk = new Tensor<float>(m, n);
            Assert.True(VulkanGemmKernel.ExecuteAmd(a, b, cVk));
            AssertMatrixClose(cVk, cCpu, 1e-3f, "VulkanGemmKernel.ExecuteAmd");
        }

        // Verify transposed GpuTarget.Amd path as well
        using var aTBase = TensorFloatExtensions.RandomUniform([k, m], -1f, 1f, seed: 1103);
        using var cTransAmd = new Tensor<float>(m, n);
        using var cTransCpu = new Tensor<float>(m, n);
        GpuAccelerator.AcceleratedMatMul(aTBase, b, cTransAmd, transposeA: true, transposeB: false, GpuTarget.Amd);
        GpuAccelerator.AcceleratedMatMul(aTBase, b, cTransCpu, transposeA: true, transposeB: false, GpuTarget.Cpu);
        AssertMatrixClose(cTransAmd, cTransCpu, 1e-3f, "AcceleratedMatMul(GpuTarget.Amd, transA)");
    }

    [Theory]
    [InlineData(128, 64, 96)]
    [InlineData(97, 65, 83)]
    public void ExecuteDualGpuGemm_UsesDistinctNvidiaAndAmdAdapters_AndMatchesCpu(int m, int k, int n)
    {
        using var a = TensorFloatExtensions.RandomUniform([m, k], -1f, 1f, seed: 1201 + m);
        using var b = TensorFloatExtensions.RandomUniform([k, n], -1f, 1f, seed: 1202 + n);
        using var cDual = new Tensor<float>(m, n);
        using var cCpu = new Tensor<float>(m, n);
        GemmKernels.MatMul(a, b, cCpu);

        bool hasDistinctAmd =
            (OperatingSystem.IsWindows() && D3D12GemmKernel.IsAmdSupported && D3D12GemmKernel.AmdVendorId == 0x1002u) ||
            (VulkanGemmKernel.IsAmdSupported && VulkanGemmKernel.AmdVendorId == 0x1002u);

        if (!GpuAccelerator.HasNvidiaGpu || !hasDistinctAmd)
        {
            Assert.False(GpuAccelerator.ExecuteDualGpuGemm(a, b, cDual));
            return;
        }

        Assert.True(GpuAccelerator.ExecuteDualGpuGemm(a, b, cDual));
        if (OperatingSystem.IsWindows() && D3D12GemmKernel.IsAmdSupported)
        {
            Assert.Equal(0x1002u, D3D12GemmKernel.AmdVendorId);
            Assert.DoesNotContain("NVIDIA", D3D12GemmKernel.AmdDeviceName, StringComparison.OrdinalIgnoreCase);
        }
        if (VulkanGemmKernel.IsAmdSupported)
        {
            Assert.Equal(0x1002u, VulkanGemmKernel.AmdVendorId);
            Assert.DoesNotContain("NVIDIA", VulkanGemmKernel.AmdDeviceName, StringComparison.OrdinalIgnoreCase);
        }

        AssertMatrixClose(cDual, cCpu, 1e-3f, $"ExecuteDualGpuGemm({m}x{k}x{n})");
    }

    [Fact]
    public void ExecuteNvidiaGemm_HostDeviceResident_ChainedCrossStream_HasNoDataRace()
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        const int n = 512;
        using var warmupA = new Tensor<float>(1, 1);
        using var warmupB = new Tensor<float>(1, 1);
        using var warmupC = new Tensor<float>(1, 1);
        if (!GpuAccelerator.ExecuteNvidiaGemm(warmupA, warmupB, warmupC)) return;

        // Populate pool with >= 2 distinct CUDA streams
        CudaExecutionContext.ClearPool();
        var c1 = CudaExecutionContext.Rent();
        var c2 = CudaExecutionContext.Rent();
        CudaExecutionContext.Return(c1);
        CudaExecutionContext.Return(c2);

        try
        {
            using var ha = TensorFloatExtensions.RandomUniform([n, n], -0.5f, 0.5f, seed: 1301);
            using var hb = TensorFloatExtensions.RandomUniform([n, n], -0.5f, 0.5f, seed: 1302);
            using var hc = new Tensor<float>(n, n);
            using var hd = TensorFloatExtensions.RandomUniform([n, n], -0.5f, 0.5f, seed: 1303);
            using var he = new Tensor<float>(n, n);

            using var refC = new Tensor<float>(n, n);
            using var refE = new Tensor<float>(n, n);
            GemmKernels.MatMul(ha, hb, refC);
            GemmKernels.MatMul(refC, hd, refE);

            ha.AllocateDeviceMemory();
            hb.AllocateDeviceMemory();
            hc.AllocateDeviceMemory();
            hd.AllocateDeviceMemory();
            he.AllocateDeviceMemory();

            ha.CopyToDevice();
            hb.CopyToDevice();
            hd.CopyToDevice();

            for (int iter = 0; iter < 5; iter++)
            {
                Assert.True(GpuAccelerator.ExecuteNvidiaGemm(ha, hb, hc));
                Assert.True(GpuAccelerator.ExecuteNvidiaGemm(hc, hd, he));
                he.CopyToHost();
                AssertMatrixClose(he, refE, 1e-2f, $"CrossStreamChainedGemm(iter={iter})");
            }
        }
        finally
        {
            CudaExecutionContext.ClearPool();
        }
    }

    [Fact]
    public void DeviceResidentTensor_Slice_ToHost_And_DownloadTo_WorkForContiguousAndStridedViews()
    {
        if (!GpuAccelerator.HasNvidiaGpu) return;

        // 1. Contiguous outer slice ([3, 1].Slice(0, 1, 1) -> [2f])
        using (var full = DeviceResidentTensor<float>.Upload([1f, 2f, 3f], 3, 1))
        using (var sub = full.Slice(0, 1, 1))
        {
            Assert.True(sub.IsContiguous);
            using var host = sub.ToHost();
            Assert.Equal(1, host.Shape[0]);
            Assert.Equal(1, host.Shape[1]);
            Assert.Equal(2f, host[0, 0]);
        }

        // 2. Multi-row contiguous outer slice ([4, 5].Slice(0, 1, 2))
        using (var hostFull = TensorFloatExtensions.RandomUniform([4, 5], -5f, 5f, seed: 1401))
        using (var devFull = DeviceResidentTensor<float>.FromHost(hostFull))
        using (var hostSlice = hostFull.Slice(0, 1, 2))
        using (var devSlice = devFull.Slice(0, 1, 2))
        {
            Assert.True(devSlice.IsContiguous);
            using var downloaded = devSlice.ToHost();
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 5; c++)
                    Assert.Equal(hostSlice[r, c], downloaded[r, c]);
        }

        // 3. Non-contiguous inner slice ([4, 8].Slice(1, 2, 3))
        using (var hostFull2d = TensorFloatExtensions.RandomUniform([4, 8], -10f, 10f, seed: 1402))
        using (var devFull2d = DeviceResidentTensor<float>.FromHost(hostFull2d))
        using (var hostInner = hostFull2d.Slice(1, 2, 3))
        using (var devInner = devFull2d.Slice(1, 2, 3))
        {
            Assert.False(devInner.IsContiguous);
            using var toHost2d = devInner.ToHost();
            using var explicitDst = new Tensor<float>(4, 3);
            devInner.DownloadTo(explicitDst);

            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    Assert.Equal(hostInner[r, c], toHost2d[r, c]);
                    Assert.Equal(hostInner[r, c], explicitDst[r, c]);
                }
            }
        }

        // 4. 3D non-contiguous slices ([2, 3, 4].Slice(1, 1, 1) and .Slice(2, 1, 2))
        using (var host3d = TensorFloatExtensions.RandomUniform([2, 3, 4], -3f, 3f, seed: 1403))
        using (var dev3d = DeviceResidentTensor<float>.FromHost(host3d))
        using (var hMid = host3d.Slice(1, 1, 1))
        using (var dMid = dev3d.Slice(1, 1, 1))
        using (var hLast = host3d.Slice(2, 1, 2))
        using (var dLast = dev3d.Slice(2, 1, 2))
        {
            Assert.False(dMid.IsContiguous);
            Assert.False(dLast.IsContiguous);
            using var gotMid = dMid.ToHost();
            using var gotLast = dLast.ToHost();

            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 4; c++)
                    Assert.Equal(hMid[b, 0, c], gotMid[b, 0, c]);

            for (int b = 0; b < 2; b++)
                for (int m = 0; m < 3; m++)
                    for (int c = 0; c < 2; c++)
                        Assert.Equal(hLast[b, m, c], gotLast[b, m, c]);
        }

        // 5. MatMul rejects non-contiguous destination slice
        using (var aDev = DeviceResidentTensor<float>.AllocateCuda(4, 4))
        using (var bDev = DeviceResidentTensor<float>.AllocateCuda(4, 3))
        using (var cBacking = DeviceResidentTensor<float>.AllocateCuda(4, 8))
        using (var cNonContig = cBacking.Slice(1, 2, 3))
        {
            Assert.False(cNonContig.IsContiguous);
            Assert.Throws<InvalidOperationException>(() => aDev.MatMul(bDev, cNonContig));
        }
    }

    private static void AssertMatrixClose(Tensor<float> actual, Tensor<float> expected, float tolerance, string label)
    {
        var act = actual.AsSpan();
        var exp = expected.AsSpan();
        Assert.Equal(exp.Length, act.Length);
        float maxErr = 0f;
        int worstIdx = -1;
        for (int i = 0; i < exp.Length; i++)
        {
            float err = MathF.Abs(act[i] - exp[i]);
            if (err > maxErr)
            {
                maxErr = err;
                worstIdx = i;
            }
        }

        Assert.True(maxErr <= tolerance,
            $"{label} max error {maxErr:E3} at index {worstIdx} exceeds tolerance {tolerance:E3} (got={act[worstIdx]}, exp={exp[worstIdx]}).");
    }
}
