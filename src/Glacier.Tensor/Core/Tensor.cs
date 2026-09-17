using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using Glacier.Gpu.Drivers;

namespace Glacier.Tensor.Core;

/// <summary>
/// High-performance N-dimensional strided tensor backed by 64-byte aligned unmanaged memory.
/// Delivers zero-allocation views (slicing, transposition, reshaping) and AVX-512 vectorization.
/// Supports device-resident GPU memory buffers (CUdeviceptr) to eliminate PCIe transfer ping-pongs.
/// </summary>
public sealed unsafe class Tensor<T> : IDisposable where T : unmanaged
{
    private static int s_idCounter;

    private readonly NativeMemoryBlock<T> _memory;
    private readonly Shape8 _shape;
    private readonly Shape8 _strides;
    private readonly long _elementOffset;
    private readonly bool _isContiguous;
    private readonly int _tensorId;
    private Tensor<T>? _grad;
    private bool _requiresGrad;
    private bool _disposed;
    private IntPtr _devicePointer;

    public int TensorId => _tensorId;
    public int Rank => _shape.Rank;
    public Shape8 Shape => _shape;
    public Shape8 Strides => _strides;
    public long ElementCount => _shape.ElementCount;
    public bool IsContiguous => _isContiguous;
    public long ElementOffset => _elementOffset;
    public NativeMemoryBlock<T> MemoryBlock => _memory;
    public IntPtr DevicePointer => _devicePointer;
    public bool IsDeviceResident => _devicePointer != IntPtr.Zero;

    public Tensor<T>? Grad
    {
        get => _grad;
        set => _grad = value;
    }

    public bool RequiresGrad
    {
        get => _requiresGrad;
        set => _requiresGrad = value;
    }

    public T* DataPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _memory.Pointer + _elementOffset;
        }
    }

    public Tensor(NativeMemoryBlock<T> memory, Shape8 shape, Shape8 strides, long elementOffset = 0)
    {
        _memory = memory;
        _shape = shape;
        _strides = strides;
        _elementOffset = elementOffset;
        _isContiguous = _shape.IsContiguous(_strides);
        _tensorId = Interlocked.Increment(ref s_idCounter);
    }

    public Tensor(in Shape8 shape)
    {
        _shape = shape;
        _strides = Shape8.ComputeContiguousStrides(_shape);
        _memory = new NativeMemoryBlock<T>(_shape.ElementCount, clear: true);
        _elementOffset = 0;
        _isContiguous = true;
        _tensorId = Interlocked.Increment(ref s_idCounter);
    }

    public Tensor(params ReadOnlySpan<int> shape) : this(new Shape8(shape))
    {
    }

    // -------------------------------------------------------------------------
    // Indexing
    // -------------------------------------------------------------------------

    public ref T this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Debug.Assert(Rank == 1);
            return ref *(DataPointer + (long)i * _strides[0]);
        }
    }

    public ref T this[int i, int j]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Debug.Assert(Rank == 2);
            return ref *(DataPointer + (long)i * _strides[0] + (long)j * _strides[1]);
        }
    }

    public ref T this[int i, int j, int k]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Debug.Assert(Rank == 3);
            return ref *(DataPointer + (long)i * _strides[0] + (long)j * _strides[1] + (long)k * _strides[2]);
        }
    }

    public ref T this[int i, int j, int k, int l]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Debug.Assert(Rank == 4);
            return ref *(DataPointer + (long)i * _strides[0] + (long)j * _strides[1] + (long)k * _strides[2] + (long)l * _strides[3]);
        }
    }

    public ref T this[ReadOnlySpan<int> indices]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (indices.Length != Rank)
                throw new ArgumentException($"Index count {indices.Length} does not match rank {Rank}");

            long offset = 0;
            for (int d = 0; d < indices.Length; d++)
            {
                offset += (long)indices[d] * _strides[d];
            }
            return ref *(DataPointer + offset);
        }
    }

    // -------------------------------------------------------------------------
    // Zero-Heap Views & Slicing
    // -------------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Slice(int dimension, int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)dimension >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(dimension));
        if (start < 0 || start + length > _shape[dimension])
            throw new ArgumentOutOfRangeException($"Invalid slice: start {start}, len {length} for dim size {_shape[dimension]}");

        long newOffset = _elementOffset + (long)start * _strides[dimension];
        Shape8 newShape = _shape.WithDimension(dimension, length);

        _memory.AddRef();
        return new Tensor<T>(_memory, newShape, _strides, newOffset);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Transpose(int dim0, int dim1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Shape8 newShape = _shape.SwapDimensions(dim0, dim1);
        Shape8 newStrides = _strides.SwapDimensions(dim0, dim1);

        _memory.AddRef();
        return new Tensor<T>(_memory, newShape, newStrides, _elementOffset);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Transpose()
    {
        if (Rank != 2)
            throw new InvalidOperationException($"Parameterless Transpose() requires a 2D tensor, but got rank {Rank}.");
        return Transpose(0, 1);
    }

    public Tensor<T> Reshape(params ReadOnlySpan<int> newShape)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var shape = new Shape8(newShape);
        if (shape.ElementCount != ElementCount)
            throw new ArgumentException($"Cannot reshape tensor of {ElementCount} elements to shape with {shape.ElementCount} elements.");

        if (IsContiguous)
        {
            _memory.AddRef();
            return new Tensor<T>(_memory, shape, Shape8.ComputeContiguousStrides(shape), _elementOffset);
        }

        // Non-contiguous: make contiguous copy first
        using var contig = Contiguous();
        return contig.Reshape(newShape);
    }

    public Tensor<T> Contiguous()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (IsContiguous && _elementOffset == 0)
        {
            _memory.AddRef();
            return new Tensor<T>(_memory, _shape, _strides, 0);
        }

        var result = new Tensor<T>(_shape);
        CopyElements(this, result);
        return result;
    }

    public Tensor<T> Clone()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var clone = new Tensor<T>(_shape);
        CopyElements(this, clone);
        return clone;
    }

    private static void CopyElements(Tensor<T> src, Tensor<T> dst)
    {
        int rank = src.Rank;
        if (rank == 1)
        {
            int n = src.Shape[0];
            for (int i = 0; i < n; i++) dst[i] = src[i];
        }
        else if (rank == 2)
        {
            int rows = src.Shape[0];
            int cols = src.Shape[1];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    dst[r, c] = src[r, c];
                }
            }
        }
        else
        {
            Span<int> indices = stackalloc int[rank];
            long count = src.ElementCount;
            for (long i = 0; i < count; i++)
            {
                dst[indices] = src[indices];
                for (int d = rank - 1; d >= 0; d--)
                {
                    indices[d]++;
                    if (indices[d] < src.Shape[d]) break;
                    indices[d] = 0;
                }
            }
        }
    }

    public Span<T> AsSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!IsContiguous)
            throw new InvalidOperationException("AsSpan() is only valid for contiguous tensors. Call Contiguous() first.");
        return new Span<T>(DataPointer, (int)ElementCount);
    }

    public void Fill(T value)
    {
        if (IsContiguous)
        {
            AsSpan().Fill(value);
        }
        else
        {
            Span<int> indices = stackalloc int[Rank];
            long count = ElementCount;
            for (long i = 0; i < count; i++)
            {
                this[indices] = value;
                for (int d = Rank - 1; d >= 0; d--)
                {
                    indices[d]++;
                    if (indices[d] < Shape[d]) break;
                    indices[d] = 0;
                }
            }
        }
    }

    public T[] ToArray()
    {
        var arr = new T[ElementCount];
        using var contig = Contiguous();
        contig.AsSpan().CopyTo(arr);
        return arr;
    }

    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------

    public static Tensor<T> Zeros(params ReadOnlySpan<int> shape)
    {
        return new Tensor<T>(new Shape8(shape));
    }

    public static Tensor<T> Zeros(in Shape8 shape)
    {
        return new Tensor<T>(shape);
    }

    public static Tensor<T> FromSpan(ReadOnlySpan<T> data, params ReadOnlySpan<int> shape)
    {
        var tensor = new Tensor<T>(shape);
        data.CopyTo(tensor.AsSpan());
        return tensor;
    }

    public static Tensor<T> FromArray(T[] data, params ReadOnlySpan<int> shape)
    {
        return FromSpan(data.AsSpan(), shape);
    }

    /// <summary>
    /// Allocates unmanaged GPU device memory (CUdeviceptr) for this tensor.
    /// </summary>
    public bool AllocateDeviceMemory()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_devicePointer != IntPtr.Zero) return true;
        if (!CuDriver.IsAvailable()) return false;

        nuint bytes = (nuint)(ElementCount * sizeof(T));
        int res = CuDriver.MemAlloc(out _devicePointer, bytes);
        return res == 0 && _devicePointer != IntPtr.Zero;
    }

    /// <summary>
    /// Copies tensor data from host memory to allocated GPU device memory.
    /// </summary>
    public bool CopyToDevice(IntPtr stream = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_devicePointer == IntPtr.Zero && !AllocateDeviceMemory())
            return false;

        nuint bytes = (nuint)(ElementCount * sizeof(T));
        fixed (T* ptr = AsSpan())
        {
            if (stream == IntPtr.Zero)
            {
                return CuDriver.MemcpyHtoD(_devicePointer, (IntPtr)ptr, bytes) == 0;
            }
            else
            {
                return CuDriver.MemcpyHtoDAsync(_devicePointer, (IntPtr)ptr, bytes, stream) == 0;
            }
        }
    }

    /// <summary>
    /// Copies tensor data from GPU device memory back to host memory.
    /// </summary>
    public bool CopyToHost(IntPtr stream = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_devicePointer == IntPtr.Zero) return false;

        nuint bytes = (nuint)(ElementCount * sizeof(T));
        fixed (T* ptr = AsSpan())
        {
            if (stream == IntPtr.Zero)
            {
                return CuDriver.MemcpyDtoH((IntPtr)ptr, _devicePointer, bytes) == 0;
            }
            else
            {
                return CuDriver.MemcpyDtoHAsync((IntPtr)ptr, _devicePointer, bytes, stream) == 0;
            }
        }
    }

    /// <summary>
    /// Frees allocated GPU device memory.
    /// </summary>
    public void FreeDeviceMemory()
    {
        if (_devicePointer != IntPtr.Zero)
        {
            CuDriver.MemFree(_devicePointer);
            _devicePointer = IntPtr.Zero;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            FreeDeviceMemory();
            _grad?.Dispose();
            _grad = null;
            _memory.Dispose();
        }
    }

    public override string ToString()
    {
        return $"Tensor<{typeof(T).Name}>(shape={_shape}, contiguous={_isContiguous})";
    }
}
