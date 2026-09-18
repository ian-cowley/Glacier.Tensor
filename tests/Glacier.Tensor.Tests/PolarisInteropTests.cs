using System;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Tensor.Core;
using Glacier.Tensor.Interop;
using Xunit;

namespace Glacier.Tensor.Tests;

public class PolarisInteropTests
{
    [Fact]
    public void DataFrame_ToTensor_ExtractsColumnsAccurately()
    {
        var col1 = new Float32Series("f1", 3);
        col1.Memory.Span[0] = 1.0f;
        col1.Memory.Span[1] = 2.0f;
        col1.Memory.Span[2] = 3.0f;

        var col2 = new Float32Series("f2", 3);
        col2.Memory.Span[0] = 10.0f;
        col2.Memory.Span[1] = 20.0f;
        col2.Memory.Span[2] = 30.0f;

        var df = new DataFrame(new ISeries[] { col1, col2 });

        using var tensor = df.ToTensor("f1", "f2");

        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(2, tensor.Shape[1]);

        Assert.Equal(1.0f, tensor[0, 0]);
        Assert.Equal(10.0f, tensor[0, 1]);
        Assert.Equal(2.0f, tensor[1, 0]);
        Assert.Equal(20.0f, tensor[1, 1]);
        Assert.Equal(3.0f, tensor[2, 0]);
        Assert.Equal(30.0f, tensor[2, 1]);
    }

    [Fact]
    public void Tensor_ToSeries_ProducesPolarisSeries()
    {
        using var t = Tensor<float>.FromArray([5.5f, 6.5f, 7.5f], 3);
        var series = t.ToSeries("my_series");

        Assert.Equal("my_series", series.Name);
        Assert.Equal(3, series.Length);
        Assert.Equal(5.5f, series.Memory.Span[0]);
        Assert.Equal(6.5f, series.Memory.Span[1]);
        Assert.Equal(7.5f, series.Memory.Span[2]);
    }

    [Fact]
    public void DataFrame_ToTensor_FastPath_SingleColumn()
    {
        int rows = 100;
        var col = new Float32Series("single_col", rows);
        for (int r = 0; r < rows; r++)
        {
            col.Memory.Span[r] = r * 1.5f;
        }

        var df = new DataFrame(new ISeries[] { col });
        using var tensor = df.ToTensor("single_col");

        Assert.Equal(rows, tensor.Shape[0]);
        Assert.Equal(1, tensor.Shape[1]);
        for (int r = 0; r < rows; r++)
        {
            Assert.Equal(r * 1.5f, tensor[r, 0]);
        }
    }

    [Fact]
    public void DataFrame_ToTensor_LargeTiledParallel_ExtractsColumnsAccurately()
    {
        int rows = 2500; // >= 2048 to trigger Parallel.For row blocks
        int cols = 70;   // > 64 to exercise both tile columns

        var seriesList = new ISeries[cols];
        for (int c = 0; c < cols; c++)
        {
            var s = new Float32Series($"c_{c}", rows);
            var span = s.Memory.Span;
            for (int r = 0; r < rows; r++)
            {
                span[r] = (float)(r * 100 + c);
            }
            seriesList[c] = s;
        }

        var df = new DataFrame(seriesList);
        using var tensor = df.ToTensor();

        Assert.Equal(rows, tensor.Shape[0]);
        Assert.Equal(cols, tensor.Shape[1]);

        // Sample checkpoint verification across all tiles
        for (int r = 0; r < rows; r += 73)
        {
            for (int c = 0; c < cols; c += 11)
            {
                float expected = (float)(r * 100 + c);
                Assert.Equal(expected, tensor[r, c]);
            }
        }
    }

    [Fact]
    public void DataFrame_ToTensor_HeterogeneousColumns_ExtractsColumnsAccurately()
    {
        int rows = 2100; // >= 2048 with heterogeneous types
        var colF32 = new Float32Series("f32", rows);
        var colF64 = new Float64Series("f64", rows);
        var colI32 = new Int32Series("i32", rows);
        var colI64 = new Int64Series("i64", rows);

        for (int r = 0; r < rows; r++)
        {
            colF32.Memory.Span[r] = r * 1.0f;
            colF64.Memory.Span[r] = r * 2.0;
            colI32.Memory.Span[r] = r * 3;
            colI64.Memory.Span[r] = r * 4L;
        }

        var df = new DataFrame(new ISeries[] { colF32, colF64, colI32, colI64 });
        using var tensor = df.ToTensor();

        Assert.Equal(rows, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]);

        for (int r = 0; r < rows; r += 50)
        {
            Assert.Equal((float)r * 1.0f, tensor[r, 0]);
            Assert.Equal((float)(r * 2.0), tensor[r, 1]);
            Assert.Equal((float)(r * 3), tensor[r, 2]);
            Assert.Equal((float)(r * 4L), tensor[r, 3]);
        }
    }
}
