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
    private const nuint MaxPooledBufferBytes = 64u * 1024u * 1024u;
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

    private int _inPool;
    private int _disposed;

    public static int PooledCount => Math.Max(0, Volatile.Read(ref s_poolCount));

    static CudaExecutionContext()
    {
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => ClearPool();
    }

    private CudaExecutionContext()
    {
        GpuAccelerator.BindNvidiaContext();
        int res = CuDriver.StreamCreate(out IntPtr stream, 0);
        if (res != 0)
        {
            throw new InvalidOperationException($"Failed to create CUDA stream, error code: {res}");
        }
        Stream = stream;
    }

    ~CudaExecutionContext()
    {
        try
        {
            DisposeCore();
        }
        catch (Exception ex)
        {
            _ = ex;
        }
    }

    public static CudaExecutionContext Rent()
    {
        GpuAccelerator.BindNvidiaContext();
        while (s_pool.TryDequeue(out var ctx))
        {
            Interlocked.Decrement(ref s_poolCount);
            if (Volatile.Read(ref ctx._disposed) == 0)
            {
                Interlocked.Exchange(ref ctx._inPool, 0);
                return ctx;
            }
        }

        return new CudaExecutionContext();
    }

    public static void Return(CudaExecutionContext? ctx)
    {
        if (ctx == null || Volatile.Read(ref ctx._disposed) != 0) return;

        // CAS guard prevents double-return pool corruption
        if (Interlocked.CompareExchange(ref ctx._inPool, 1, 0) != 0)
        {
            return;
        }

        // Release oversized per-stream buffers before returning to pool
        ctx.ShrinkOversizedBuffers();

        if (Interlocked.Increment(ref s_poolCount) <= MaxPoolSize)
        {
            s_pool.Enqueue(ctx);
        }
        else
        {
            Interlocked.Decrement(ref s_poolCount);
            Interlocked.Exchange(ref ctx._inPool, 0);
            ctx.Dispose();
        }
    }

    public static void TrimPool(int maxRetained = 0)
    {
        if (maxRetained < 0) maxRetained = 0;
        while (Volatile.Read(ref s_poolCount) > maxRetained && s_pool.TryDequeue(out var ctx))
        {
            Interlocked.Decrement(ref s_poolCount);
            Interlocked.Exchange(ref ctx._inPool, 0);
            ctx.Dispose();
        }
    }

    public static void ClearPool() => TrimPool(0);

    private void ShrinkOversizedBuffers()
    {
        if (CapA <= MaxPooledBufferBytes && CapB <= MaxPooledBufferBytes && CapC <= MaxPooledBufferBytes)
            return;

        GpuAccelerator.BindNvidiaContext();
        if (CapA > MaxPooledBufferBytes && DevA != IntPtr.Zero)
        {
            CuDriver.MemFree(DevA);
            DevA = IntPtr.Zero;
            CapA = 0;
        }
        if (CapB > MaxPooledBufferBytes && DevB != IntPtr.Zero)
        {
            CuDriver.MemFree(DevB);
            DevB = IntPtr.Zero;
            CapB = 0;
        }
        if (CapC > MaxPooledBufferBytes && DevC != IntPtr.Zero)
        {
            CuDriver.MemFree(DevC);
            DevC = IntPtr.Zero;
            CapC = 0;
        }
    }

    private static nuint ComputeGrowthBytes(nuint currentCap, nuint requiredBytes)
    {
        nuint grown;
        try
        {
            grown = checked(currentCap + currentCap / 2);
        }
        catch (OverflowException)
        {
            grown = requiredBytes;
        }
        return grown > requiredBytes ? grown : requiredBytes;
    }

    public void EnsureCapacityA(nuint requiredBytes)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (requiredBytes > CapA)
        {
            GpuAccelerator.BindNvidiaContext();
            nuint allocBytes = ComputeGrowthBytes(CapA, requiredBytes);

            if (DevA != IntPtr.Zero)
            {
                int freeRes = CuDriver.MemFree(DevA);
                DevA = IntPtr.Zero;
                CapA = 0;
                if (freeRes != 0)
                    CuDriver.Check(freeRes, "cuMemFree(DevA)");
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
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (requiredBytes > CapB)
        {
            GpuAccelerator.BindNvidiaContext();
            nuint allocBytes = ComputeGrowthBytes(CapB, requiredBytes);

            if (DevB != IntPtr.Zero)
            {
                int freeRes = CuDriver.MemFree(DevB);
                DevB = IntPtr.Zero;
                CapB = 0;
                if (freeRes != 0)
                    CuDriver.Check(freeRes, "cuMemFree(DevB)");
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
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (requiredBytes > CapC)
        {
            GpuAccelerator.BindNvidiaContext();
            nuint allocBytes = ComputeGrowthBytes(CapC, requiredBytes);

            if (DevC != IntPtr.Zero)
            {
                int freeRes = CuDriver.MemFree(DevC);
                DevC = IntPtr.Zero;
                CapC = 0;
                if (freeRes != 0)
                    CuDriver.Check(freeRes, "cuMemFree(DevC)");
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

    private void DisposeCore()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

        GpuAccelerator.BindNvidiaContext();
        if (DevA != IntPtr.Zero)
        {
            _ = CuDriver.MemFree(DevA);
            DevA = IntPtr.Zero;
            CapA = 0;
        }
        if (DevB != IntPtr.Zero)
        {
            _ = CuDriver.MemFree(DevB);
            DevB = IntPtr.Zero;
            CapB = 0;
        }
        if (DevC != IntPtr.Zero)
        {
            _ = CuDriver.MemFree(DevC);
            DevC = IntPtr.Zero;
            CapC = 0;
        }
        if (Stream != IntPtr.Zero)
        {
            _ = CuDriver.StreamDestroy(Stream);
            Stream = IntPtr.Zero;
        }
    }

    public void Dispose()
    {
        DisposeCore();
        GC.SuppressFinalize(this);
    }
}
