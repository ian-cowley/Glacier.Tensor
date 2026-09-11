using System;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

public class TensorStridedTests
{
    [Fact]
    public void Shape8_ComputesContiguousStridesAccurately()
    {
        var shape = Shape8.Create(2, 3, 4);
        var strides = Shape8.ComputeContiguousStrides(shape);

        Assert.Equal(3, shape.Rank);
        Assert.Equal(24, shape.ElementCount);
        Assert.Equal(12, strides[0]);
        Assert.Equal(4, strides[1]);
        Assert.Equal(1, strides[2]);
        Assert.True(shape.IsContiguous(strides));
    }

    [Fact]
    public void Tensor_ZerosAndIndexing_WorkCorrectly()
    {
        using var t = Tensor<float>.Zeros(3, 4);
        Assert.Equal(2, t.Rank);
        Assert.Equal(12, t.ElementCount);
        Assert.True(t.IsContiguous);

        t[1, 2] = 42.0f;
        Assert.Equal(42.0f, t[1, 2]);
        Assert.Equal(0.0f, t[0, 0]);
        Assert.Equal(0.0f, t[2, 3]);
    }

    [Fact]
    public void Tensor_Slice_CreatesZeroCopyView()
    {
        using var t = new Tensor<float>(4, 5);
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 5; c++)
            {
                t[r, c] = r * 10 + c;
            }
        }

        // Slice rows 1..2 (length 2)
        using var slice = t.Slice(0, 1, 2);
        Assert.Equal(2, slice.Shape[0]);
        Assert.Equal(5, slice.Shape[1]);

        Assert.Equal(10.0f, slice[0, 0]);
        Assert.Equal(14.0f, slice[0, 4]);
        Assert.Equal(20.0f, slice[1, 0]);
        Assert.Equal(24.0f, slice[1, 4]);

        // Mutating slice mutates underlying memory
        slice[0, 0] = 999.0f;
        Assert.Equal(999.0f, t[1, 0]);
    }

    [Fact]
    public void Tensor_Transpose_SwapsDimensionsWithoutCopy()
    {
        using var t = new Tensor<float>(2, 3);
        t[0, 0] = 1.0f; t[0, 1] = 2.0f; t[0, 2] = 3.0f;
        t[1, 0] = 4.0f; t[1, 1] = 5.0f; t[1, 2] = 6.0f;

        using var tT = t.Transpose();
        Assert.Equal(3, tT.Shape[0]);
        Assert.Equal(2, tT.Shape[1]);

        Assert.Equal(1.0f, tT[0, 0]);
        Assert.Equal(4.0f, tT[0, 1]);
        Assert.Equal(2.0f, tT[1, 0]);
        Assert.Equal(5.0f, tT[1, 1]);
        Assert.Equal(3.0f, tT[2, 0]);
        Assert.Equal(6.0f, tT[2, 1]);
    }

    [Fact]
    public void Tensor_Reshape_And_Contiguous_Succeeds()
    {
        using var t = TensorFloatExtensions.Ones(2, 6);
        using var reshaped = t.Reshape(3, 4);

        Assert.Equal(3, reshaped.Shape[0]);
        Assert.Equal(4, reshaped.Shape[1]);
        Assert.Equal(12, reshaped.ElementCount);
        Assert.Equal(1.0f, reshaped[2, 3]);
    }
}
