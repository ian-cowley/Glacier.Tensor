// <copyright file="DeviceResidentTensorTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Tensor.Tests;

using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Xunit;

public class DeviceResidentTensorTests
{
    private static bool IsAnyGpuAvailable => CuDriver.IsAvailable() || (OperatingSystem.IsWindows() && D3D12GemmKernel.IsSupported);

    [Fact]
    public void Allocate_SetsShapeAndContiguousStrides()
    {
        if (!IsAnyGpuAvailable) return;

        using var tensor = DeviceResidentTensor<float>.Allocate(4, 8, 16);

        Assert.Equal(3, tensor.Rank);
        Assert.Equal(4, tensor.Shape[0]);
        Assert.Equal(8, tensor.Shape[1]);
        Assert.Equal(16, tensor.Shape[2]);
        Assert.Equal(4 * 8 * 16, tensor.ElementCount);
        Assert.True(tensor.IsContiguous);
        Assert.Equal(0, tensor.ElementOffset);
        Assert.False(tensor.IsDisposed);
    }

    [Fact]
    public void Upload_And_ToHost_Roundtrip()
    {
        if (!IsAnyGpuAvailable) return;

        int[] shape = [16, 32];
        int count = 16 * 32;
        float[] hostData = new float[count];
        for (int i = 0; i < count; i++)
        {
            hostData[i] = (float)(i * 0.25 - 10.0);
        }

        using var resident = DeviceResidentTensor<float>.Upload(hostData, shape);

        Assert.Equal(count, resident.ElementCount);
        Assert.Equal(2, resident.Rank);

        using var downloaded = resident.ToHost();

        Assert.Equal(count, downloaded.ElementCount);
        var downloadedSpan = downloaded.AsSpan();
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(hostData[i], downloadedSpan[i]);
        }
    }

    [Fact]
    public void FromHost_And_DownloadTo_PreservesValues()
    {
        if (!IsAnyGpuAvailable) return;

        using var hostSource = TensorFloatExtensions.RandomUniform([64, 32], -5f, 5f, seed: 42);
        using var resident = DeviceResidentTensor<float>.FromHost(hostSource);

        using var hostDest = new Tensor<float>(64, 32);
        resident.DownloadTo(hostDest);

        var srcSpan = hostSource.AsSpan();
        var dstSpan = hostDest.AsSpan();
        for (int i = 0; i < hostSource.ElementCount; i++)
        {
            Assert.Equal(srcSpan[i], dstSpan[i]);
        }
    }

    [Fact]
    public void View_ReshapesContiguousTensorWithoutDataCopy()
    {
        if (!IsAnyGpuAvailable) return;

        using var original = DeviceResidentTensor<float>.Allocate(2, 4, 8);
        var newShape = new Shape8([8, 8]);
        using var reshaped = original.View(newShape);

        Assert.Equal(2, reshaped.Rank);
        Assert.Equal(8, reshaped.Shape[0]);
        Assert.Equal(8, reshaped.Shape[1]);
        Assert.Equal(original.ElementCount, reshaped.ElementCount);
        Assert.Equal(original.DevicePointer, reshaped.DevicePointer);
        Assert.Equal(original.GpuVirtualAddress, reshaped.GpuVirtualAddress);
    }

    [Fact]
    public void Slice_AdjustsShapeAndElementOffset()
    {
        if (!IsAnyGpuAvailable) return;

        using var tensor = DeviceResidentTensor<float>.Allocate(10, 20);
        using var slice = tensor.Slice(0, 3, 4);

        Assert.Equal(2, slice.Rank);
        Assert.Equal(4, slice.Shape[0]);
        Assert.Equal(20, slice.Shape[1]);
        Assert.Equal(3 * 20, slice.ElementOffset);
    }

    [Fact]
    public void MatMul_ZeroStagingResidentGemm_MatchesCpuGemm()
    {
        if (!IsAnyGpuAvailable) return;

        int M = 128, K = 96, N = 64;
        using var aHost = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 1234);
        using var bHost = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 5678);

        // Compute ground truth on CPU
        using var cExpected = new Tensor<float>(M, N);
        GemmKernels.MatMul(aHost, bHost, cExpected);

        // Upload to resident VRAM
        using var aDev = DeviceResidentTensor<float>.FromHost(aHost);
        using var bDev = DeviceResidentTensor<float>.FromHost(bHost);

        // Execute resident GEMM
        using var cDev = aDev.MatMul(bDev);

        // Download result and verify numerical accuracy
        using var cActual = cDev.ToHost();

        var expSpan = cExpected.AsSpan();
        var actSpan = cActual.AsSpan();

        float maxDiff = 0f;
        for (int i = 0; i < cExpected.ElementCount; i++)
        {
            float diff = MathF.Abs(expSpan[i] - actSpan[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Assert.True(maxDiff < 1e-3f, $"Max difference {maxDiff} exceeded tolerance 1e-3f");
    }

    [Fact]
    public void MatMul_ChainedOperationsInVram_WithoutIntermediateHostDownloads()
    {
        if (!IsAnyGpuAvailable) return;

        int size = 64;
        using var aDev = DeviceResidentTensor<float>.Allocate(size, size);
        using var bDev = DeviceResidentTensor<float>.Allocate(size, size);
        using var cDev = DeviceResidentTensor<float>.Allocate(size, size);

        float[] initA = new float[size * size];
        float[] initB = new float[size * size];
        Array.Fill(initA, 1.0f / size);
        Array.Fill(initB, 1.0f);

        aDev.Buffer.CopyFromHost(initA);
        bDev.Buffer.CopyFromHost(initB);
        cDev.Buffer.CopyFromHost(initB);

        // Chain 1: H1 = A * B in VRAM
        using var h1 = aDev.MatMul(bDev);
        Assert.False(h1.IsDisposed);
        Assert.Equal(size, h1.Shape[0]);
        Assert.Equal(size, h1.Shape[1]);

        // Chain 2: H2 = H1 * C in VRAM (zero PCIe roundtrip)
        using var h2 = h1.MatMul(cDev);
        Assert.False(h2.IsDisposed);

        using var finalHost = h2.ToHost();
        var span = finalHost.AsSpan();
        Assert.True(span[0] > 0f);
    }

    [Fact]
    public void Dispose_GuardsPropertiesAndMethods()
    {
        if (!IsAnyGpuAvailable) return;

        var tensor = DeviceResidentTensor<float>.Allocate(8, 8);
        tensor.Dispose();

        Assert.True(tensor.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => _ = tensor.DevicePointer);
        Assert.Throws<ObjectDisposedException>(() => _ = tensor.GpuVirtualAddress);
        Assert.Throws<ObjectDisposedException>(() => tensor.ToHost());
    }

    [Fact]
    public void MatMul_ShapeMismatches_ThrowArgumentException()
    {
        if (!IsAnyGpuAvailable) return;

        using var a = DeviceResidentTensor<float>.Allocate(10, 20);
        using var b = DeviceResidentTensor<float>.Allocate(30, 40); // 20 != 30

        Assert.Throws<ArgumentException>(() => a.MatMul(b));
    }
}
