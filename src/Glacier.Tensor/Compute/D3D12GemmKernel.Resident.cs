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
        diagnostics = null;
        D3D12ExecutionStage stage = D3D12ExecutionStage.ArgumentValidation;
        ulong bytesA = (ulong)(m * k * sizeof(float));
        ulong bytesB = (ulong)(k * n * sizeof(float));
        ulong bytesC = (ulong)(m * n * sizeof(float));
        uint gridX = (uint)((n + 63) / 64);
        uint gridY = (uint)((m + 63) / 64);

        if (gpuVaA == 0 || gpuVaB == 0 || gpuVaC == 0)
        {
            diagnostics = RecordError(stage, new ArgumentException("GPU Virtual Addresses must be non-zero."), m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
            return false;
        }

        try
        {
            stage = D3D12ExecutionStage.Initialization;
            if (!EnsureInitialized())
            {
                diagnostics = RecordError(stage, new InvalidOperationException("Direct3D 12 initialization failed or device not available on this platform."), m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
                return false;
            }

            lock (s_lock)
            {
                stage = D3D12ExecutionStage.CommandRecording;
                s_cmdAlloc!.Reset();
                s_cmdList!.Reset(s_cmdAlloc, null);

                // Bind compute root signature and pipeline
                s_cmdList.SetComputeRootSignature(s_rootSig!);
                s_cmdList.SetPipelineState(s_pipelineState!);

                uint* pConsts = stackalloc uint[4];
                pConsts[0] = (uint)m;
                pConsts[1] = (uint)k;
                pConsts[2] = (uint)n;
                pConsts[3] = 0;
                s_cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

                // Direct root descriptors to VRAM buffers
                s_cmdList.SetComputeRootShaderResourceView(1, gpuVaA);
                s_cmdList.SetComputeRootShaderResourceView(2, gpuVaB);
                s_cmdList.SetComputeRootUnorderedAccessView(3, gpuVaC);

                stage = D3D12ExecutionStage.Dispatch;
                s_cmdList.Dispatch(gridX, gridY, 1);



                stage = D3D12ExecutionStage.QueueExecution;
                s_cmdList.Close();
                s_queue!.ExecuteCommandList(s_cmdList);

                stage = D3D12ExecutionStage.FenceSynchronization;
                Synchronize();

                t_lastError = null;
                diagnostics = new D3D12KernelDiagnostics
                {
                    Success = true,
                    Stage = D3D12ExecutionStage.None,
                    DeviceName = DeviceName,
                    M = m,
                    K = k,
                    N = n,
                    BytesA = bytesA,
                    BytesB = bytesB,
                    BytesC = bytesC,
                    GridX = gridX,
                    GridY = gridY,
                    Timestamp = DateTimeOffset.UtcNow
                };

                return true;
            }
        }
        catch (Exception ex)
        {
            diagnostics = RecordError(stage, ex, m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
            return false;
        }
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
