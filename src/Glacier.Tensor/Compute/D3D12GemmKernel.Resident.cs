// <copyright file="D3D12GemmKernel.Resident.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Tensor.Compute;

using System;
using Glacier.Tensor.Core;
using Glacier.Tensor.Diagnostics;
using Vortice.Direct3D12;

public static unsafe partial class D3D12GemmKernel
{
    /// <summary>
    /// Executes GEMM directly between device-resident GPU virtual addresses via Direct3D 12 compute.
    /// Eliminates all host upload and readback PCIe staging overhead.
    /// </summary>
    public static bool ExecuteResident(ulong gpuVaA, ulong gpuVaB, ulong gpuVaC, int m, int k, int n)
    {
        return ExecuteResident(gpuVaA, gpuVaB, gpuVaC, m, k, n, out _);
    }

    /// <summary>
    /// Executes GEMM directly between device-resident GPU virtual addresses with diagnostic reporting.
    /// </summary>
    public static bool ExecuteResident(
        ulong gpuVaA,
        ulong gpuVaB,
        ulong gpuVaC,
        int m,
        int k,
        int n,
        out D3D12KernelDiagnostics? diagnostics)
    {
        return s_default.ExecuteResident(gpuVaA, gpuVaB, gpuVaC, m, k, n, out diagnostics);
    }

    /// <summary>
    /// Executes GEMM directly between device-resident tensors via Direct3D 12 compute.
    /// </summary>
    public static bool ExecuteResident(DeviceResidentTensor<float> a, DeviceResidentTensor<float> b, DeviceResidentTensor<float> c)
    {
        return ExecuteResident(a, b, c, out _);
    }

    /// <summary>
    /// Executes GEMM directly between device-resident tensors with diagnostic reporting.
    /// </summary>
    public static bool ExecuteResident(
        DeviceResidentTensor<float> a,
        DeviceResidentTensor<float> b,
        DeviceResidentTensor<float> c,
        out D3D12KernelDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(a, nameof(a));
        ArgumentNullException.ThrowIfNull(b, nameof(b));
        ArgumentNullException.ThrowIfNull(c, nameof(c));

        if (a.Rank != 2 || b.Rank != 2 || c.Rank != 2)
            throw new ArgumentException("Matrices must be Rank 2.");
        if (!a.IsContiguous || !b.IsContiguous || !c.IsContiguous)
            throw new InvalidOperationException("Direct3D 12 resident GEMM requires contiguous tensor layouts.");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException($"Inner dimension mismatch: {a.Shape[1]} != {b.Shape[0]}");
        if (c.Shape[0] != a.Shape[0] || c.Shape[1] != b.Shape[1])
            throw new ArgumentException("Destination dimensions do not match product shape.");

        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[1];

        return ExecuteResident(a.GpuVirtualAddress, b.GpuVirtualAddress, c.GpuVirtualAddress, m, k, n, out diagnostics);
    }
}
