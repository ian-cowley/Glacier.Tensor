using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Glacier.Tensor.Core;

/// <summary>
/// Fixed-size value-type buffer holding up to 8 64-bit element strides on the CPU stack.
/// </summary>
[InlineArray(8)]
public struct Strides8Buffer
{
    private long _element0;
}

/// <summary>
/// Value-type N-dimensional 64-bit stride descriptor (rank &lt;= 8) with zero heap allocation.
/// Prevents 32-bit stride overflow on tensors with more than 2^31 - 1 elements.
/// </summary>
public readonly struct Strides8 : IEquatable<Strides8>
{
    private readonly Strides8Buffer _strides;
    private readonly int _rank;

    public int Rank => _rank;

    public long this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if ((uint)index >= (uint)_rank)
                throw new IndexOutOfRangeException($"Stride index {index} out of bounds for rank {_rank}");
            return _strides[index];
        }
    }

    public Strides8(ReadOnlySpan<long> strides)
    {
        if (strides.Length > 8)
            throw new ArgumentException("Glacier.Tensor supports tensors up to rank 8.", nameof(strides));

        _rank = strides.Length;
        _strides = default;
        for (int i = 0; i < strides.Length; i++)
        {
            if (strides[i] < 0)
                throw new ArgumentException($"Stride must be non-negative. Got {strides[i]} at index {i}.", nameof(strides));
            _strides[i] = strides[i];
        }
    }

    private Strides8(Strides8Buffer buffer, int rank)
    {
        _strides = buffer;
        _rank = rank;
    }

    public static Strides8 Create(params ReadOnlySpan<long> strides) => new Strides8(strides);

    public static Strides8 FromShape8(in Shape8 strides)
    {
        int rank = strides.Rank;
        if (rank == 0) return default;

        Strides8Buffer buffer = default;
        for (int i = 0; i < rank; i++)
        {
            buffer[i] = strides[i];
        }
        return new Strides8(buffer, rank);
    }

    public static Strides8 ComputeContiguous(in Shape8 shape)
    {
        int rank = shape.Rank;
        if (rank == 0) return default;

        Strides8Buffer strides = default;
        strides[rank - 1] = 1L;
        for (int i = rank - 2; i >= 0; i--)
        {
            strides[i] = checked(strides[i + 1] * shape[i + 1]);
        }
        return new Strides8(strides, rank);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Strides8 WithStride(int dimension, long stride)
    {
        if ((uint)dimension >= (uint)_rank)
            throw new ArgumentOutOfRangeException(nameof(dimension));
        if (stride < 0)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be non-negative.");

        Strides8Buffer copy = _strides;
        copy[dimension] = stride;
        return new Strides8(copy, _rank);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Strides8 SwapDimensions(int dim0, int dim1)
    {
        if ((uint)dim0 >= (uint)_rank || (uint)dim1 >= (uint)_rank)
            throw new ArgumentOutOfRangeException("Dimension index out of range for rank.");

        Strides8Buffer copy = _strides;
        long tmp = copy[dim0];
        copy[dim0] = copy[dim1];
        copy[dim1] = tmp;
        return new Strides8(copy, _rank);
    }

    public bool IsContiguous(in Shape8 shape) => shape.IsContiguous(in this);

    public Shape8 ToShape8()
    {
        if (_rank == 0) return default;
        Span<int> dims = stackalloc int[_rank];
        for (int i = 0; i < _rank; i++)
        {
            dims[i] = checked((int)_strides[i]);
        }
        return new Shape8(dims);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<long> AsSpan()
    {
        if (_rank == 0) return ReadOnlySpan<long>.Empty;
        return MemoryMarshal.CreateReadOnlySpan(ref Unsafe.AsRef(in _strides[0]), _rank);
    }

    public void CopyTo(Span<long> destination)
    {
        for (int i = 0; i < _rank && i < destination.Length; i++)
        {
            destination[i] = _strides[i];
        }
    }

    public long[] ToArray()
    {
        var arr = new long[_rank];
        for (int i = 0; i < _rank; i++)
        {
            arr[i] = _strides[i];
        }
        return arr;
    }

    public bool Equals(Strides8 other)
    {
        if (_rank != other._rank) return false;
        for (int i = 0; i < _rank; i++)
        {
            if (_strides[i] != other._strides[i]) return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is Strides8 other && Equals(other);

    public override int GetHashCode()
    {
        HashCode hc = default;
        hc.Add(_rank);
        for (int i = 0; i < _rank; i++)
        {
            hc.Add(_strides[i]);
        }
        return hc.ToHashCode();
    }

    public static bool operator ==(Strides8 left, Strides8 right) => left.Equals(right);
    public static bool operator !=(Strides8 left, Strides8 right) => !left.Equals(right);

    public override string ToString()
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < _rank; i++)
        {
            sb.Append(_strides[i]);
            if (i < _rank - 1) sb.Append(", ");
        }
        sb.Append(']');
        return sb.ToString();
    }
}
