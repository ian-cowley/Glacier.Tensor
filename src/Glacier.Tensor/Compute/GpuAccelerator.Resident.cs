// <copyright file="GpuAccelerator.Resident.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Tensor.Compute;

using System;
using Glacier.Gpu.Drivers;
using Glacier.Tensor.Core;

public static unsafe partial class GpuAccelerator
{
    /// <summary>
    /// Executes GEMM directly between device-resident GPU tensors via NVIDIA dGPU SASS tiled kernel.
    /// Operates entirely within VRAM with zero host transfer staging overhead.
    /// </summary>
    public static bool ExecuteNvidiaGemm(
        DeviceResidentTensor<float> a,
        DeviceResidentTensor<float> b,
        DeviceResidentTensor<float> c,
        IntPtr stream = default)
    {
        ArgumentNullException.ThrowIfNull(a, nameof(a));
        ArgumentNullException.ThrowIfNull(b, nameof(b));
        ArgumentNullException.ThrowIfNull(c, nameof(c));

        if (a.Rank != 2 || b.Rank != 2 || c.Rank != 2)
            throw new ArgumentException("All tensors must be Rank 2 matrices.");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException($"Inner dimension mismatch: {a.Shape[1]} != {b.Shape[0]}");
        if (c.Shape[0] != a.Shape[0] || c.Shape[1] != b.Shape[1])
            throw new ArgumentException($"Result tensor shape [{c.Shape[0]}, {c.Shape[1]}] does not match product dimensions [{a.Shape[0]}, {b.Shape[1]}].");

        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[1];

        return ExecuteNvidiaGemm(a.DevicePointer, b.DevicePointer, c.DevicePointer, m, n, k, stream);
    }

    /// <summary>
    /// Executes GEMM directly over raw NVIDIA CUDA device pointers via SASS tiled kernel.
    /// </summary>
    public static bool ExecuteNvidiaGemm(
        IntPtr devA,
        IntPtr devB,
        IntPtr devC,
        int m,
        int n,
        int k,
        IntPtr stream = default)
    {
        if (!EnsureNvidiaInitialized() || s_cuGemmFn == IntPtr.Zero)
            return false;
        if (devA == IntPtr.Zero || devB == IntPtr.Zero || devC == IntPtr.Zero)
            return false;

        CuDriver.CtxSetCurrent(s_cuContext);

        IntPtr actualStream = stream;
        CudaExecutionContext? ctx = null;
        if (actualStream == IntPtr.Zero)
        {
            try
            {
                ctx = CudaExecutionContext.Rent();
                actualStream = ctx.Stream;
            }
            catch
            {
                return false;
            }
        }

        try
        {
            void** kParams = stackalloc void*[6];
            kParams[0] = &devA;
            kParams[1] = &devB;
            kParams[2] = &devC;
            kParams[3] = &m;
            kParams[4] = &n;
            kParams[5] = &k;

            uint gX = (uint)((n + 63) / 64);
            uint gY = (uint)((m + 63) / 64);

            int lRes = CuDriver.LaunchKernel(
                s_cuGemmFn,
                gX, gY, 1,
                16, 16, 1,
                0, actualStream,
                (IntPtr)kParams,
                IntPtr.Zero
            );

            if (lRes != 0) return false;
            CuDriver.StreamSynchronize(actualStream);
            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            if (ctx != null)
            {
                CudaExecutionContext.Return(ctx);
            }
        }
    }

    /// <summary>
    /// Executes GEMM directly between device-resident tensors on NVIDIA Tensor Cores (RTX 4060) via WMMA hardware instructions.
    /// </summary>
    public static bool ExecuteNvidiaTensorCoreGemm(
        DeviceResidentTensor<float> a,
        DeviceResidentTensor<float> b,
        DeviceResidentTensor<float> c,
        IntPtr stream = default)
    {
        ArgumentNullException.ThrowIfNull(a, nameof(a));
        ArgumentNullException.ThrowIfNull(b, nameof(b));
        ArgumentNullException.ThrowIfNull(c, nameof(c));

        if (a.Rank != 2 || b.Rank != 2 || c.Rank != 2)
            throw new ArgumentException("All tensors must be Rank 2 matrices.");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException($"Inner dimension mismatch: {a.Shape[1]} != {b.Shape[0]}");
        if (c.Shape[0] != a.Shape[0] || c.Shape[1] != b.Shape[1])
            throw new ArgumentException($"Result tensor shape [{c.Shape[0]}, {c.Shape[1]}] does not match product dimensions [{a.Shape[0]}, {b.Shape[1]}].");

        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[1];

        return ExecuteNvidiaTensorCoreGemm(a.DevicePointer, b.DevicePointer, c.DevicePointer, m, n, k, stream);
    }

    /// <summary>
    /// Executes GEMM directly over raw NVIDIA CUDA device pointers on Tensor Cores via WMMA hardware instructions.
    /// </summary>
    public static bool ExecuteNvidiaTensorCoreGemm(
        IntPtr devA,
        IntPtr devB,
        IntPtr devC,
        int m,
        int n,
        int k,
        IntPtr stream = default)
    {
        if (!EnsureNvidiaInitialized() || s_cuTensorCoreFp32Fn == IntPtr.Zero)
            return false;
        if (devA == IntPtr.Zero || devB == IntPtr.Zero || devC == IntPtr.Zero)
            return false;

        CuDriver.CtxSetCurrent(s_cuContext);

        IntPtr actualStream = stream;
        CudaExecutionContext? ctx = null;
        if (actualStream == IntPtr.Zero)
        {
            try
            {
                ctx = CudaExecutionContext.Rent();
                actualStream = ctx.Stream;
            }
            catch
            {
                return false;
            }
        }

        try
        {
            void** kParams = stackalloc void*[6];
            kParams[0] = &devA;
            kParams[1] = &devB;
            kParams[2] = &devC;
            kParams[3] = &m;
            kParams[4] = &n;
            kParams[5] = &k;

            uint gX = (uint)((m + 15) / 16);
            uint gY = (uint)((n + 15) / 16);

            int lRes = CuDriver.LaunchKernel(
                s_cuTensorCoreFp32Fn,
                gX, gY, 1,
                32, 1, 1,
                0, actualStream,
                (IntPtr)kParams,
                IntPtr.Zero
            );

            if (lRes != 0) return false;
            CuDriver.StreamSynchronize(actualStream);
            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            if (ctx != null)
            {
                CudaExecutionContext.Return(ctx);
            }
        }
    }
}
