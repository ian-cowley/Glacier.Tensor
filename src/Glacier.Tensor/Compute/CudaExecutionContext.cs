using System;
using System.Collections.Concurrent;
using System.Threading;
using Glacier.Gpu.Drivers;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Thread-safe execution context maintaining a dedicated CUDA stream and per-stream device buffers.
/// Eliminates global synchronization locks and enables multi-threaded concurrent GPU GEMMs.
/// </summary>
public sealed class CudaExecutionContext : IDisposable
{
    private static readonly ConcurrentQueue<CudaExecutionContext> s_pool = new();
    private static int s_poolCount;
    private static readonly int MaxPoolSize = Math.Max(32, Environment.ProcessorCount * 2);

    public IntPtr Stream { get; private set; }
    public IntPtr DevA { get; private set; }
    public IntPtr DevB { get; private set; }
    public IntPtr DevC { get; private set; }
    public nuint CapA { get; private set; }
    public nuint CapB { get; private set; }
    public nuint CapC { get; private set; }

    private bool _disposed;

    private CudaExecutionContext()
    {
        int res = CuDriver.StreamCreate(out IntPtr stream, 0);
        if (res != 0)
        {
            throw new InvalidOperationException($"Failed to create CUDA stream, error code: {res}");
        }
        Stream = stream;
    }

    public static CudaExecutionContext Rent()
    {
        while (s_pool.TryDequeue(out var ctx))
        {
            Interlocked.Decrement(ref s_poolCount);
            if (!ctx._disposed)
            {
                return ctx;
            }
        }

        return new CudaExecutionContext();
    }

    public static void Return(CudaExecutionContext? ctx)
    {
        if (ctx == null || ctx._disposed) return;

        if (Interlocked.Increment(ref s_poolCount) <= MaxPoolSize)
        {
            s_pool.Enqueue(ctx);
        }
        else
        {
            Interlocked.Decrement(ref s_poolCount);
            ctx.Dispose();
        }
    }

    public void EnsureCapacityA(nuint requiredBytes)
    {
        if (requiredBytes > CapA)
        {
            nuint allocBytes = Math.Max(requiredBytes, (nuint)(CapA * 1.5));
            if (allocBytes < requiredBytes) allocBytes = requiredBytes;

            if (DevA != IntPtr.Zero)
            {
                CuDriver.MemFree(DevA);
                DevA = IntPtr.Zero;
                CapA = 0;
            }

            if (CuDriver.MemAlloc(out IntPtr ptr, allocBytes) != 0)
            {
                if (CuDriver.MemAlloc(out ptr, requiredBytes) != 0)
                {
                    throw new OutOfMemoryException($"Failed to allocate {requiredBytes} bytes on CUDA device for matrix A.");
                }
                allocBytes = requiredBytes;
            }
            DevA = ptr;
            CapA = allocBytes;
        }
    }

    public void EnsureCapacityB(nuint requiredBytes)
    {
        if (requiredBytes > CapB)
        {
            nuint allocBytes = Math.Max(requiredBytes, (nuint)(CapB * 1.5));
            if (allocBytes < requiredBytes) allocBytes = requiredBytes;

            if (DevB != IntPtr.Zero)
            {
                CuDriver.MemFree(DevB);
                DevB = IntPtr.Zero;
                CapB = 0;
            }

            if (CuDriver.MemAlloc(out IntPtr ptr, allocBytes) != 0)
            {
                if (CuDriver.MemAlloc(out ptr, requiredBytes) != 0)
                {
                    throw new OutOfMemoryException($"Failed to allocate {requiredBytes} bytes on CUDA device for matrix B.");
                }
                allocBytes = requiredBytes;
            }
            DevB = ptr;
            CapB = allocBytes;
        }
    }

    public void EnsureCapacityC(nuint requiredBytes)
    {
        if (requiredBytes > CapC)
        {
            nuint allocBytes = Math.Max(requiredBytes, (nuint)(CapC * 1.5));
            if (allocBytes < requiredBytes) allocBytes = requiredBytes;

            if (DevC != IntPtr.Zero)
            {
                CuDriver.MemFree(DevC);
                DevC = IntPtr.Zero;
                CapC = 0;
            }

            if (CuDriver.MemAlloc(out IntPtr ptr, allocBytes) != 0)
            {
                if (CuDriver.MemAlloc(out ptr, requiredBytes) != 0)
                {
                    throw new OutOfMemoryException($"Failed to allocate {requiredBytes} bytes on CUDA device for matrix C.");
                }
                allocBytes = requiredBytes;
            }
            DevC = ptr;
            CapC = allocBytes;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (DevA != IntPtr.Zero)
        {
            CuDriver.MemFree(DevA);
            DevA = IntPtr.Zero;
            CapA = 0;
        }
        if (DevB != IntPtr.Zero)
        {
            CuDriver.MemFree(DevB);
            DevB = IntPtr.Zero;
            CapB = 0;
        }
        if (DevC != IntPtr.Zero)
        {
            CuDriver.MemFree(DevC);
            DevC = IntPtr.Zero;
            CapC = 0;
        }
        if (Stream != IntPtr.Zero)
        {
            CuDriver.StreamDestroy(Stream);
            Stream = IntPtr.Zero;
        }
    }
}
