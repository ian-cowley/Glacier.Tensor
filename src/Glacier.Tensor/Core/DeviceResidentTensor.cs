// <copyright file="DeviceResidentTensor.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Tensor.Core;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Threading;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Tensor.Compute;
using Vortice.Direct3D12;

/// <summary>
/// N-dimensional strided tensor residing exclusively in GPU device memory (VRAM).
/// Bypasses CPU heap allocation entirely and enables zero-staging multi-TFLOPS chained GPU compute.
/// </summary>
/// <typeparam name="T">Unmanaged data type.</typeparam>
public sealed unsafe class DeviceResidentTensor<T> : IDisposable where T : unmanaged
{
    private static int s_idCounter;

    private sealed class BufferRefCount
    {
        public int Count = 1;
    }

    private readonly GpuBuffer<T> _buffer;
    private readonly BufferRefCount _refCount;
    private readonly Shape8 _shape;
    private readonly Strides8 _longStrides;
    private readonly long _elementOffset;
    private readonly bool _isContiguous;
    private readonly int _tensorId;
    private int _disposed;

    /// <summary>Gets the unique tensor identifier.</summary>
    public int TensorId => _tensorId;

    /// <summary>Gets the number of dimensions.</summary>
    public int Rank => _shape.Rank;

    /// <summary>Gets the N-dimensional shape.</summary>
    public Shape8 Shape => _shape;

    /// <summary>Gets the 64-bit stride vector.</summary>
    public Strides8 LongStrides => _longStrides;

    /// <summary>Gets the stride vector.</summary>
    public Shape8 Strides => _longStrides.ToShape8();

    /// <summary>Gets the total number of elements.</summary>
    public long ElementCount => _shape.ElementCount;

    /// <summary>Gets a value indicating whether tensor elements are contiguous in device memory.</summary>
    public bool IsContiguous => _isContiguous;

    /// <summary>Gets the element offset within the underlying GPU buffer.</summary>
    public long ElementOffset => _elementOffset;

    /// <summary>Gets the underlying GPU device buffer.</summary>
    public GpuBuffer<T> Buffer => _buffer;

    /// <summary>Gets a value indicating whether this tensor has been disposed.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>Gets the hardware GPU classification type.</summary>
    public GpuDeviceType DeviceType => _buffer.DeviceType;

    /// <summary>Gets the raw device pointer adjusted for element offset.</summary>
    public IntPtr DevicePointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _buffer.DevicePointer == IntPtr.Zero
                ? IntPtr.Zero
                : _buffer.DevicePointer + (nint)checked(_elementOffset * (long)sizeof(T));
        }
    }

    /// <summary>Gets the 64-bit GPU Virtual Address adjusted for element offset.</summary>
    public ulong GpuVirtualAddress
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _buffer.GpuVirtualAddress == 0
                ? 0
                : _buffer.GpuVirtualAddress + checked((ulong)_elementOffset * (ulong)sizeof(T));
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DeviceResidentTensor{T}"/> class with explicit strides and offset.
    /// </summary>
    public DeviceResidentTensor(GpuBuffer<T> buffer, Shape8 shape, Shape8 strides, long elementOffset = 0)
        : this(buffer, new BufferRefCount(), shape, Strides8.FromShape8(strides), elementOffset, incrementRef: false)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DeviceResidentTensor{T}"/> class with explicit 64-bit strides and offset.
    /// </summary>
    public DeviceResidentTensor(GpuBuffer<T> buffer, Shape8 shape, Strides8 strides, long elementOffset = 0)
        : this(buffer, new BufferRefCount(), shape, strides, elementOffset, incrementRef: false)
    {
    }

    private DeviceResidentTensor(GpuBuffer<T> buffer, BufferRefCount refCount, Shape8 shape, Strides8 strides, long elementOffset, bool incrementRef)
    {
        _buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _refCount = refCount ?? throw new ArgumentNullException(nameof(refCount));
        if (incrementRef)
        {
            Interlocked.Increment(ref _refCount.Count);
        }
        _shape = shape;
        _longStrides = strides;
        _elementOffset = elementOffset;
        _isContiguous = _shape.IsContiguous(in _longStrides);
        _tensorId = Interlocked.Increment(ref s_idCounter);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DeviceResidentTensor{T}"/> class with contiguous strides.
    /// </summary>
    public DeviceResidentTensor(GpuBuffer<T> buffer, in Shape8 shape)
        : this(buffer, shape, Strides8.ComputeContiguous(shape), 0)
    {
    }

    /// <summary>
    /// Allocates an uninitialized device-resident tensor in GPU VRAM with zero host memory overhead.
    /// Automatically selects the active hardware backend (CUDA on NVIDIA, Direct3D 12 on Windows).
    /// </summary>
    public static DeviceResidentTensor<T> Allocate(params ReadOnlySpan<int> shape)
    {
        var sh = new Shape8(shape);
        var strides = Strides8.ComputeContiguous(sh);
        int elemCount = checked((int)sh.ElementCount);

        GpuBuffer<T> buffer;
        if (GpuAccelerator.BindNvidiaContext())
        {
            buffer = GpuBuffer<T>.AllocateCuda(elemCount);
        }
        else if (OperatingSystem.IsWindows() && D3D12GemmKernel.IsSupported && D3D12GemmKernel.Device != null)
        {
            buffer = GpuBuffer<T>.AllocateD3D12(D3D12GemmKernel.Device, elemCount, ResourceFlags.AllowUnorderedAccess);
        }
        else
        {
            throw new PlatformNotSupportedException("No supported GPU device runtime (CUDA or Direct3D 12) available for resident tensor allocation.");
        }

        return new DeviceResidentTensor<T>(buffer, sh, strides, 0);
    }

    /// <summary>
    /// Allocates an uninitialized device-resident tensor in NVIDIA CUDA VRAM.
    /// </summary>
    public static DeviceResidentTensor<T> AllocateCuda(params ReadOnlySpan<int> shape)
    {
        GpuAccelerator.BindNvidiaContext();
        var sh = new Shape8(shape);
        var strides = Strides8.ComputeContiguous(sh);
        var buffer = GpuBuffer<T>.AllocateCuda(checked((int)sh.ElementCount));
        return new DeviceResidentTensor<T>(buffer, sh, strides, 0);
    }

    [SupportedOSPlatform("windows")]
    public static DeviceResidentTensor<T> AllocateD3D12(ID3D12Device device, ReadOnlySpan<int> shape, ResourceFlags flags = ResourceFlags.AllowUnorderedAccess)
    {
        var sh = new Shape8(shape);
        var strides = Strides8.ComputeContiguous(sh);
        var buffer = GpuBuffer<T>.AllocateD3D12(device, checked((int)sh.ElementCount), flags);
        return new DeviceResidentTensor<T>(buffer, sh, strides, 0);
    }

    /// <summary>
    /// Allocates an uninitialized device-resident tensor on a Direct3D 12 device default heap.
    /// </summary>
    [SupportedOSPlatform("windows")]
    public static DeviceResidentTensor<T> AllocateD3D12(ID3D12Device device, params int[] shape)
        => AllocateD3D12(device, shape.AsSpan(), ResourceFlags.AllowUnorderedAccess);

    /// <summary>
    /// Allocates device VRAM and uploads host data in a single decoupled operation.
    /// </summary>
    public static DeviceResidentTensor<T> Upload(ReadOnlySpan<T> hostData, params ReadOnlySpan<int> shape)
    {
        var tensor = Allocate(shape);
        tensor._buffer.CopyFromHost(hostData);
        return tensor;
    }

    /// <summary>
    /// Converts a host <see cref="Tensor{T}"/> into a device-resident tensor.
    /// </summary>
    public static DeviceResidentTensor<T> FromHost(Tensor<T> hostTensor, IntPtr stream = default)
    {
        ArgumentNullException.ThrowIfNull(hostTensor, nameof(hostTensor));
        if (!hostTensor.IsContiguous)
            throw new InvalidOperationException("Device resident tensors require contiguous source data for direct upload.");

        var tensor = Allocate(hostTensor.Shape.AsSpan());
        tensor._buffer.CopyFromHost(hostTensor.AsSpan(), stream);
        return tensor;
    }

    /// <summary>
    /// Downloads GPU device data back to an existing host <see cref="Tensor{T}"/>.
    /// </summary>
    public void DownloadTo(Tensor<T> destination, IntPtr stream = default)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ObjectDisposedException.ThrowIf(_buffer.IsDisposed, _buffer);
        ArgumentNullException.ThrowIfNull(destination, nameof(destination));
        if (destination.ElementCount != ElementCount)
            throw new ArgumentException($"Destination tensor element count ({destination.ElementCount}) does not match device tensor ({ElementCount}).");
        if (!destination.IsContiguous)
            throw new InvalidOperationException("Destination tensor must be contiguous for direct device download.");

        if (ElementCount == 0)
        {
            return;
        }

        if (_isContiguous && _elementOffset == 0 && ElementCount == _buffer.ElementCount)
        {
            _buffer.CopyToHost(destination.AsSpan(), stream);
            return;
        }

        if (_buffer.DeviceType != GpuDeviceType.NvidiaDiscrete)
        {
            throw new NotSupportedException($"Direct host download for sliced or offset views is not supported for device type {_buffer.DeviceType}.");
        }

        if (!CuDriver.IsAvailable())
        {
            throw new PlatformNotSupportedException("NVIDIA CUDA driver is unavailable.");
        }

        GpuBuffer<T>.EnsureCudaContext();

        IntPtr baseDevPtr = _buffer.DevicePointer;
        if (baseDevPtr == IntPtr.Zero)
        {
            throw new InvalidOperationException("Underlying CUDA device pointer is null.");
        }

        Span<T> dstSpan = destination.AsSpan();
        if (_isContiguous)
        {
            if (_elementOffset < 0 || checked(_elementOffset + ElementCount) > _buffer.ElementCount)
            {
                throw new ArgumentOutOfRangeException(nameof(_elementOffset), "Sliced device tensor range exceeds underlying buffer bounds.");
            }

            nuint byteCount = checked((nuint)ElementCount * (nuint)sizeof(T));
            fixed (T* pDst = dstSpan)
            {
                CopyDeviceToHost((IntPtr)pDst, DevicePointer, byteCount, stream);
            }
            return;
        }

        int rank = _shape.Rank;
        long innerStride = rank > 0 ? _longStrides[rank - 1] : 1L;
        int innerLen = rank > 0 ? _shape[rank - 1] : 1;
        bool innerContiguous = innerStride == 1;
        int runElements = innerContiguous ? innerLen : 1;
        nuint runBytes = checked((nuint)runElements * (nuint)sizeof(T));
        int outerDimCount = innerContiguous ? rank - 1 : rank;

        Span<int> coords = stackalloc int[Math.Max(1, outerDimCount)];
        coords.Clear();

        long outerIterations = 1;
        for (int d = 0; d < outerDimCount; d++)
        {
            outerIterations = checked(outerIterations * _shape[d]);
        }

        fixed (T* pDstBase = dstSpan)
        {
            long dstIdx = 0;
            for (long iter = 0; iter < outerIterations; iter++)
            {
                long srcElemOffset = _elementOffset;
                for (int d = 0; d < outerDimCount; d++)
                {
                    srcElemOffset = checked(srcElemOffset + (long)coords[d] * _longStrides[d]);
                }

                if (srcElemOffset < 0 || checked(srcElemOffset + runElements) > _buffer.ElementCount)
                {
                    throw new ArgumentOutOfRangeException(nameof(_elementOffset), "Strided device tensor slice exceeds underlying buffer bounds.");
                }

                IntPtr srcPtr = baseDevPtr + (nint)checked(srcElemOffset * (long)sizeof(T));
                IntPtr dstPtr = (IntPtr)(pDstBase + dstIdx);
                CopyDeviceToHost(dstPtr, srcPtr, runBytes, stream);
                dstIdx += runElements;

                for (int d = outerDimCount - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < _shape[d])
                    {
                        break;
                    }
                    coords[d] = 0;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CopyDeviceToHost(IntPtr dstHost, IntPtr srcDevice, nuint byteCount, IntPtr stream)
    {
        if (stream == IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.MemcpyDtoH(dstHost, srcDevice, byteCount), "cuMemcpyDtoH");
        }
        else
        {
            CuDriver.Check(CuDriver.MemcpyDtoHAsync(dstHost, srcDevice, byteCount, stream), "cuMemcpyDtoHAsync");
        }
    }

    /// <summary>
    /// Creates a host <see cref="Tensor{T}"/> and copies VRAM contents into it across PCIe.
    /// </summary>
    public Tensor<T> ToHost(IntPtr stream = default)
    {
        var host = new Tensor<T>(_shape);
        DownloadTo(host, stream);
        return host;
    }

    /// <summary>
    /// Creates a reshaped view over the device resident memory without copying data.
    /// </summary>
    public DeviceResidentTensor<T> View(in Shape8 newShape)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        if (!_isContiguous)
            throw new InvalidOperationException("Cannot reshape a non-contiguous device-resident tensor view.");
        if (newShape.ElementCount != ElementCount)
            throw new ArgumentException($"Cannot reshape tensor with {ElementCount} elements to shape with {newShape.ElementCount} elements.");

        return new DeviceResidentTensor<T>(_buffer, _refCount, newShape, Strides8.ComputeContiguous(newShape), _elementOffset, incrementRef: true);
    }

    /// <summary>
    /// Slices this tensor along the specified dimension, creating a sub-view without copying data.
    /// </summary>
    public DeviceResidentTensor<T> Slice(int dimension, int start, int length)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        if ((uint)dimension >= (uint)_shape.Rank)
            throw new ArgumentOutOfRangeException(nameof(dimension));
        if (start < 0 || length < 0 || (long)start + length > _shape[dimension])
            throw new ArgumentOutOfRangeException(nameof(start), "Slice range is out of bounds.");

        Span<int> newDims = stackalloc int[_shape.Rank];
        _shape.CopyTo(newDims);
        newDims[dimension] = length;

        long newOffset = checked(_elementOffset + (long)start * _longStrides[dimension]);
        var newShape = new Shape8(newDims);
        return new DeviceResidentTensor<T>(_buffer, _refCount, newShape, _longStrides, newOffset, incrementRef: true);
    }

    /// <summary>
    /// Chains resident matrix multiplication in pure VRAM without host roundtrip staging.
    /// Achieves multi-TFLOPS throughput on NVIDIA and DirectX 12 hardware.
    /// </summary>
    public DeviceResidentTensor<float> MatMul(DeviceResidentTensor<float> other, DeviceResidentTensor<float>? destination = null, IntPtr stream = default)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentNullException.ThrowIfNull(other, nameof(other));
        ObjectDisposedException.ThrowIf(other.IsDisposed, other);

        if (typeof(T) != typeof(float))
            throw new NotSupportedException("MatMul is only supported for float tensors.");

        var a = (DeviceResidentTensor<float>)(object)this;
        var b = other;

        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMul requires Rank 2 matrices.");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException($"Inner dimension mismatch: {a.Shape[1]} != {b.Shape[0]}");

        if (!a.IsContiguous || !b.IsContiguous)
            throw new InvalidOperationException("DeviceResidentTensor.MatMul requires contiguous tensors. Sliced or strided views must be copied to a contiguous buffer prior to GEMM execution.");

        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[1];

        if (destination != null)
        {
            ObjectDisposedException.ThrowIf(destination.IsDisposed, destination);
            if (destination.Rank != 2)
                throw new ArgumentException("MatMul destination must be a Rank 2 matrix.", nameof(destination));
            if (destination.Shape[0] != m || destination.Shape[1] != n)
                throw new ArgumentException($"Destination tensor shape [{destination.Shape[0]}, {destination.Shape[1]}] does not match product dimensions [{m}, {n}].", nameof(destination));
            if (!destination.IsContiguous)
                throw new InvalidOperationException("DeviceResidentTensor.MatMul requires a contiguous destination tensor.");
        }
        else
        {
            destination = DeviceResidentTensor<float>.Allocate(m, n);
        }

        // 1. NVIDIA CUDA fast resident GEMM path
        if (a.DevicePointer != IntPtr.Zero && b.DevicePointer != IntPtr.Zero && destination.DevicePointer != IntPtr.Zero)
        {
            if (GpuAccelerator.ExecuteNvidiaGemm(a, b, destination, stream))
            {
                return destination;
            }
        }

        // 2. Direct3D 12 resident GEMM path (AMD Radeon APUs / discrete GPUs on Windows)
        if (OperatingSystem.IsWindows() && D3D12GemmKernel.IsSupported &&
            a.GpuVirtualAddress != 0 && b.GpuVirtualAddress != 0 && destination.GpuVirtualAddress != 0)
        {
            if (D3D12GemmKernel.ExecuteResident(a, b, destination))
            {
                return destination;
            }
        }

        throw new InvalidOperationException("Resident GPU GEMM kernel execution failed across all supported hardware backends.");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            if (Interlocked.Decrement(ref _refCount.Count) == 0)
            {
                _buffer.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"DeviceResidentTensor<{typeof(T).Name}>(shape={_shape}, device={_buffer.DeviceType})";
    }
}
