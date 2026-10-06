using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Glacier.Tensor.Core;

/// <summary>
/// Reference-counted 64-byte aligned unmanaged memory block.
/// Eliminates GC overhead and guarantees AVX-512 cache-line alignment.
/// </summary>
public sealed unsafe class NativeMemoryBlock<T> : IDisposable where T : unmanaged
{
    private T* _pointer;
    private readonly long _elementCount;
    private readonly nuint _byteLength;
    private readonly bool _ownsMemory;
    private readonly IDisposable? _lifetimeOwner;
    private int _refCount;
    private bool _disposed;

    public T* Pointer => _pointer;
    public long ElementCount => _elementCount;
    public nuint ByteLength => _byteLength;
    public bool IsDisposed => _disposed;

    public NativeMemoryBlock(long elementCount, bool clear = true)
    {
        if (elementCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount));

        _elementCount = elementCount;
        _byteLength = (nuint)(elementCount * sizeof(T));
        _ownsMemory = true;
        _lifetimeOwner = null;
        _refCount = 1;

        if (_byteLength > 0)
        {
            _pointer = (T*)NativeMemory.AlignedAlloc(_byteLength, 64);
            if (clear)
            {
                NativeMemory.Clear(_pointer, _byteLength);
            }
        }
        else
        {
            _pointer = null;
        }
    }

    public NativeMemoryBlock(T* externalPointer, long elementCount, bool ownsMemory = false, IDisposable? lifetimeOwner = null)
    {
        if (elementCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount));

        _pointer = externalPointer;
        _elementCount = elementCount;
        _byteLength = (nuint)(elementCount * sizeof(T));
        _ownsMemory = ownsMemory;
        _lifetimeOwner = lifetimeOwner;
        _refCount = 1;
    }

    public void AddRef()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Interlocked.Increment(ref _refCount);
    }

    public Span<T> AsSpan(long offset = 0, int length = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(offset, nameof(offset));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(offset, _elementCount, nameof(offset));

        int len;
        if (length == -1)
        {
            long remaining = _elementCount - offset;
            if (remaining > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(length), remaining, "Remaining element count exceeds maximum Span<T> length.");
            }
            len = (int)remaining;
        }
        else
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length, nameof(length));
            if (length > _elementCount - offset)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, "Requested length exceeds available element capacity.");
            }
            len = length;
        }

        if (_pointer == null || len == 0) return Span<T>.Empty;

        return new Span<T>(_pointer + offset, len);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (Interlocked.Decrement(ref _refCount) == 0)
            {
                _disposed = true;
                if (_ownsMemory && _pointer != null)
                {
                    NativeMemory.AlignedFree(_pointer);
                    _pointer = null;
                }
                _lifetimeOwner?.Dispose();
            }
        }
    }
}
