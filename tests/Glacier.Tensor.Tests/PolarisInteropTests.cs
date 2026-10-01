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

    [Fact]
    public void Series_AsTensorView_ZeroCopy_ModificationsReflectImmediately()
    {
        int length = 100;
        var series = new Float32Series("s_f32", length);
        for (int i = 0; i < length; i++)
        {
            series.Memory.Span[i] = i * 1.5f;
        }

        using var tensor = series.AsTensorView();

        Assert.Equal(1, tensor.Rank);
        Assert.Equal(length, tensor.ElementCount);
        Assert.True(tensor.IsContiguous);

        // Verify initial contents
        for (int i = 0; i < length; i++)
        {
            Assert.Equal(i * 1.5f, tensor[i]);
        }

        // Zero-copy assertion: mutating series memory reflects in tensor
        series.Memory.Span[5] = 42.5f;
        Assert.Equal(42.5f, tensor[5]);

        // Zero-copy assertion: mutating tensor reflects in series memory
        tensor[10] = 99.25f;
        Assert.Equal(99.25f, series.Memory.Span[10]);
    }

    [Fact]
    public void Series_AsTensorView_2DColumnVector_ZeroCopy()
    {
        int length = 50;
        var series = new Float32Series("s_col", length);
        for (int i = 0; i < length; i++)
        {
            series.Memory.Span[i] = i * 2.0f;
        }

        using var tensor = series.AsTensorView(as2DColumn: true);

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(length, tensor.Shape[0]);
        Assert.Equal(1, tensor.Shape[1]);

        series.Memory.Span[7] = 777.0f;
        Assert.Equal(777.0f, tensor[7, 0]);

        tensor[12, 0] = 888.0f;
        Assert.Equal(888.0f, series.Memory.Span[12]);
    }

    [Fact]
    public void DataFrame_ToTensorView_SingleColumn_ZeroCopy()
    {
        int rows = 40;
        var colA = new Float32Series("col_a", rows);
        for (int i = 0; i < rows; i++)
        {
            colA.Memory.Span[i] = i * 3.0f;
        }

        var df = new DataFrame(new ISeries[] { colA });
        using var tensor = df.ToTensorView("col_a");

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(rows, tensor.Shape[0]);
        Assert.Equal(1, tensor.Shape[1]);

        colA.Memory.Span[3] = 123.4f;
        Assert.Equal(123.4f, tensor[3, 0]);

        tensor[4, 0] = 567.8f;
        Assert.Equal(567.8f, colA.Memory.Span[4]);
    }

    [Fact]
    public void Series_AsTensorView_Int32_ZeroCopy()
    {
        int length = 30;
        var series = new Int32Series("s_i32", length);
        for (int i = 0; i < length; i++)
        {
            series.Memory.Span[i] = i * 10;
        }

        using var tensor = series.AsTensorView();

        Assert.Equal(1, tensor.Rank);
        Assert.Equal(length, tensor.ElementCount);

        series.Memory.Span[0] = 12345;
        Assert.Equal(12345, tensor[0]);

        tensor[1] = 54321;
        Assert.Equal(54321, series.Memory.Span[1]);
    }
}
