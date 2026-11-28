using System;
using System.Runtime.CompilerServices;
using System.Text;

namespace Glacier.Tensor.Core;

/// <summary>
/// Fixed-size value-type buffer holding up to 8 dimension lengths or strides on the CPU stack.
/// Zero heap allocations during slicing, transposition, view operations, and broadcasting.
/// </summary>
[InlineArray(8)]
public struct Shape8Buffer
{
    private int _element0;
}

/// <summary>
/// Value-type N-dimensional shape descriptor (rank &lt;= 8) with zero heap allocation.
/// </summary>
public readonly struct Shape8 : IEquatable<Shape8>
{
    private readonly Shape8Buffer _dims;
    private readonly int _rank;

    public int Rank => _rank;

    public int this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if ((uint)index >= (uint)_rank)
                throw new IndexOutOfRangeException($"Dimension index {index} out of bounds for rank {_rank}");
            return _dims[index];
        }
    }

    public long ElementCount
    {
        get
        {
            if (_rank == 0) return 1;
            long count = 1;
            for (int i = 0; i < _rank; i++)
            {
                count = checked(count * _dims[i]);
            }
            return count;
        }
    }

    public Shape8(ReadOnlySpan<int> dims)
    {
        if (dims.Length > 8)
            throw new ArgumentException("Glacier.Tensor supports tensors up to rank 8.", nameof(dims));

        _rank = dims.Length;
        _dims = default;
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] < 0)
                throw new ArgumentException($"Dimension length must be non-negative. Got {dims[i]} at index {i}.");
            _dims[i] = dims[i];
        }
    }

    private Shape8(Shape8Buffer buffer, int rank)
    {
        _dims = buffer;
        _rank = rank;
    }

    public static Shape8 Create(params ReadOnlySpan<int> dims) => new Shape8(dims);

    public static Shape8 ComputeContiguousStrides(in Shape8 shape)
    {
        int rank = shape.Rank;
        if (rank == 0) return default;

        Shape8Buffer strides = default;
        long currentStride = 1;
        strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
        {
            currentStride = checked(currentStride * shape[i + 1]);
            strides[i] = checked((int)currentStride);
        }
        return new Shape8(strides, rank);
    }

    public static Strides8 ComputeContiguousStridesLong(in Shape8 shape)
        => Strides8.ComputeContiguous(in shape);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Shape8 WithDimension(int dimension, int length)
    {
        if ((uint)dimension >= (uint)_rank)
            throw new ArgumentOutOfRangeException(nameof(dimension));

        Shape8Buffer copy = _dims;
        copy[dimension] = length;
        return new Shape8(copy, _rank);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Shape8 SwapDimensions(int dim0, int dim1)
    {
        if ((uint)dim0 >= (uint)_rank || (uint)dim1 >= (uint)_rank)
            throw new ArgumentOutOfRangeException("Dimension index out of range for rank.");

        Shape8Buffer copy = _dims;
        int tmp = copy[dim0];
        copy[dim0] = copy[dim1];
        copy[dim1] = tmp;
        return new Shape8(copy, _rank);
    }

    public bool IsContiguous(in Shape8 strides)
    {
        if (_rank != strides.Rank) return false;
        if (_rank == 0) return true;
        long expectedStride = 1;
        for (int i = _rank - 1; i >= 0; i--)
        {
            if (strides[i] != expectedStride) return false;
            expectedStride = checked(expectedStride * _dims[i]);
        }
        return true;
    }

    public bool IsContiguous(in Strides8 strides)
    {
        if (_rank != strides.Rank) return false;
        if (_rank == 0) return true;
        long expectedStride = 1;
        for (int i = _rank - 1; i >= 0; i--)
        {
            if (strides[i] != expectedStride) return false;
            expectedStride = checked(expectedStride * _dims[i]);
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<int> AsSpan()
    {
        if (_rank == 0) return ReadOnlySpan<int>.Empty;
        return System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(ref Unsafe.AsRef(in _dims[0]), _rank);
    }

    public void CopyTo(Span<int> destination)
    {
        for (int i = 0; i < _rank && i < destination.Length; i++)
        {
            destination[i] = _dims[i];
        }
    }

    public int[] ToArray()
    {
        var arr = new int[_rank];
        for (int i = 0; i < _rank; i++)
        {
            arr[i] = _dims[i];
        }
        return arr;
    }

    public bool Equals(Shape8 other)
    {
        if (_rank != other._rank) return false;
        for (int i = 0; i < _rank; i++)
        {
            if (_dims[i] != other._dims[i]) return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is Shape8 other && Equals(other);

    public override int GetHashCode()
    {
        HashCode hc = default;
        hc.Add(_rank);
        for (int i = 0; i < _rank; i++)
        {
            hc.Add(_dims[i]);
        }
        return hc.ToHashCode();
    }

    public static bool operator ==(Shape8 left, Shape8 right) => left.Equals(right);
    public static bool operator !=(Shape8 left, Shape8 right) => !left.Equals(right);

    public override string ToString()
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < _rank; i++)
        {
            sb.Append(_dims[i]);
            if (i < _rank - 1) sb.Append(", ");
        }
        sb.Append(']');
        return sb.ToString();
    }
}
