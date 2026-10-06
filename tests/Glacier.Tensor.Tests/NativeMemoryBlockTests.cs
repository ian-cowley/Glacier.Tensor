using System;
using System.Runtime.InteropServices;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

public unsafe class NativeMemoryBlockTests
{
    [Theory]
    [InlineData(-1L)]
    [InlineData(-10L)]
    [InlineData(long.MinValue)]
    public void AsSpan_NegativeOffset_ThrowsArgumentOutOfRangeException(long negativeOffset)
    {
        using var block = new NativeMemoryBlock<float>(100);
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(negativeOffset));
        Assert.Equal("offset", ex.ParamName);
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(-100)]
    [InlineData(int.MinValue)]
    public void AsSpan_NegativeLength_ThrowsArgumentOutOfRangeException(int negativeLength)
    {
        using var block = new NativeMemoryBlock<float>(100);
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(0, negativeLength));
        Assert.Equal("length", ex.ParamName);
    }

    [Theory]
    [InlineData(101L)]
    [InlineData(1000L)]
    [InlineData(long.MaxValue)]
    public void AsSpan_OffsetExceedsCapacity_ThrowsArgumentOutOfRangeException(long excessiveOffset)
    {
        using var block = new NativeMemoryBlock<float>(100);
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(excessiveOffset, 0));
        Assert.Equal("offset", ex.ParamName);
    }

    [Fact]
    public void AsSpan_LengthExceedsCapacity_ThrowsArgumentOutOfRangeException()
    {
        using var block = new NativeMemoryBlock<float>(100);

        // Requested 15 elements from offset 90: 90 + 15 = 105 > 100
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(90, 15));
        Assert.Equal("length", ex.ParamName);

        // Requested 101 elements from offset 0: 0 + 101 = 101 > 100
        var ex2 = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(0, 101));
        Assert.Equal("length", ex2.ParamName);
    }

    [Fact]
    public void AsSpan_ArithmeticOverflowImmunity_GuardsAgainstWrapAround()
    {
        using var block = new NativeMemoryBlock<float>(100);

        // offset near long.MaxValue would wrap if checked with offset + length > count
        var exOffset = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(long.MaxValue - 2, 10));
        Assert.Equal("offset", exOffset.ParamName);

        // length near int.MaxValue from positive offset
        var exLength = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(10, int.MaxValue));
        Assert.Equal("length", exLength.ParamName);
    }

    [Fact]
    public void AsSpan_DisposedBlock_ThrowsObjectDisposedException()
    {
        var block = new NativeMemoryBlock<float>(50);
        block.Dispose();

        Assert.True(block.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => block.AsSpan());
        Assert.Throws<ObjectDisposedException>(() => block.AddRef());
    }

    [Fact]
    public void AsSpan_ZeroLengthBlock_ValidatesBoundsAccurately()
    {
        using var block = new NativeMemoryBlock<float>(0);

        // Offset 0 and length 0 or -1 on a 0-capacity block should cleanly yield Empty
        var spanDefault = block.AsSpan();
        Assert.True(spanDefault.IsEmpty);

        var spanZeroLen = block.AsSpan(0, 0);
        Assert.True(spanZeroLen.IsEmpty);

        // Out-of-bounds on zero-capacity block must strictly throw
        var exPositiveOffset = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(1, 0));
        Assert.Equal("offset", exPositiveOffset.ParamName);

        var exNegativeOffset = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(-1, 0));
        Assert.Equal("offset", exNegativeOffset.ParamName);

        var exPositiveLength = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(0, 1));
        Assert.Equal("length", exPositiveLength.ParamName);
    }

    [Fact]
    public void AsSpan_ValidSlices_ReturnAccurateSpans()
    {
        using var block = new NativeMemoryBlock<float>(100);
        var fullSpan = block.AsSpan();
        Assert.Equal(100, fullSpan.Length);

        // Populate with known values
        for (int i = 0; i < 100; i++)
        {
            fullSpan[i] = i * 2.5f;
        }

        // Subslice in interior
        var subSpan = block.AsSpan(10, 20);
        Assert.Equal(20, subSpan.Length);
        Assert.Equal(10 * 2.5f, subSpan[0]);
        Assert.Equal(29 * 2.5f, subSpan[19]);

        // Edge slice at boundary
        var edgeSpan = block.AsSpan(99, 1);
        Assert.Equal(1, edgeSpan.Length);
        Assert.Equal(99 * 2.5f, edgeSpan[0]);

        // Zero-length slice at end boundary
        var zeroEndSpan = block.AsSpan(100, 0);
        Assert.True(zeroEndSpan.IsEmpty);

        // Default remainder slice from offset 80
        var remSpan = block.AsSpan(80, -1);
        Assert.Equal(20, remSpan.Length);
        Assert.Equal(80 * 2.5f, remSpan[0]);
        Assert.Equal(99 * 2.5f, remSpan[19]);
    }

    [Fact]
    public void Constructor_NegativeElementCount_ThrowsArgumentOutOfRangeException()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => new NativeMemoryBlock<float>(-5));
        Assert.Equal("elementCount", ex.ParamName);
    }

    [Fact]
    public void ExternalPointerConstructor_NegativeElementCount_ThrowsArgumentOutOfRangeException()
    {
        float* pDummy = (float*)0x1000;
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => new NativeMemoryBlock<float>(pDummy, -10));
        Assert.Equal("elementCount", ex.ParamName);
    }

    [Fact]
    public void ExternalPointerConstructor_ValidPointer_OperatesCorrectly()
    {
        float* rawBuffer = (float*)NativeMemory.Alloc(10, sizeof(float));
        try
        {
            for (int i = 0; i < 10; i++)
            {
                rawBuffer[i] = i + 100.0f;
            }

            using var block = new NativeMemoryBlock<float>(rawBuffer, 10, ownsMemory: false);
            Assert.Equal(10, block.ElementCount);
            Assert.False(block.IsDisposed);

            var span = block.AsSpan();
            Assert.Equal(10, span.Length);
            Assert.Equal(100.0f, span[0]);
            Assert.Equal(109.0f, span[9]);

            var slice = block.AsSpan(5, 3);
            Assert.Equal(3, slice.Length);
            Assert.Equal(105.0f, slice[0]);
            Assert.Equal(107.0f, slice[2]);
        }
        finally
        {
            NativeMemory.Free(rawBuffer);
        }
    }
}
