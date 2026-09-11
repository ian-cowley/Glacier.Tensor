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

    public NativeMemoryBlock(T* externalPointer, long elementCount, bool ownsMemory = false)
    {
        _pointer = externalPointer;
        _elementCount = elementCount;
        _byteLength = (nuint)(elementCount * sizeof(T));
        _ownsMemory = ownsMemory;
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
        if (_pointer == null) return Span<T>.Empty;

        int len = length >= 0 ? length : (int)(_elementCount - offset);
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
            }
        }
    }
}
