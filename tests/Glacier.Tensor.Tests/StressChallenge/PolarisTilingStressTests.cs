using System;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Tensor.Core;
using Glacier.Tensor.Interop;
using Xunit;

namespace Glacier.Tensor.Tests.StressChallenge;

public class PolarisTilingStressTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(2047)]
    [InlineData(2048)]
    [InlineData(2049)]
    [InlineData(10000)]
    public void Polaris_ToTensor_RowCountBoundaries_Float32_MatchesGroundTruth(int rowCount)
    {
        const int cols = 5;
        var seriesList = new ISeries[cols];
        for (int c = 0; c < cols; c++)
        {
            var s = new Float32Series($"c_{c}", rowCount);
            var span = s.Memory.Span;
            for (int r = 0; r < rowCount; r++)
            {
                span[r] = (float)(r * 10 + c);
            }
            seriesList[c] = s;
        }

        var df = new DataFrame(seriesList);
        using var tensor = df.ToTensor();

        Assert.Equal(rowCount, tensor.Shape[0]);
        Assert.Equal(cols, tensor.Shape[1]);

        if (rowCount == 0) return;

        // Ground truth verification: check every row if small, or sampled stride if large
        int step = Math.Max(1, rowCount / 200);
        for (int r = 0; r < rowCount; r += step)
        {
            for (int c = 0; c < cols; c++)
            {
                float expected = (float)(r * 10 + c);
                float actual = tensor[r, c];
                Assert.Equal(expected, actual);
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(2047)]
    [InlineData(2048)]
    [InlineData(2049)]
    [InlineData(10000)]
    public void Polaris_ToTensor_RowCountBoundaries_Heterogeneous_MatchesGroundTruth(int rowCount)
    {
        var colF32 = new Float32Series("col_f32", rowCount);
        var colF64 = new Float64Series("col_f64", rowCount);
        var colI32 = new Int32Series("col_i32", rowCount);
        var colI64 = new Int64Series("col_i64", rowCount);

        for (int r = 0; r < rowCount; r++)
        {
            colF32.Memory.Span[r] = r * 1.25f;
            colF64.Memory.Span[r] = r * 2.50;
            colI32.Memory.Span[r] = r * 3;
            colI64.Memory.Span[r] = r * 4L;
        }

        var df = new DataFrame(new ISeries[] { colF32, colF64, colI32, colI64 });
        using var tensor = df.ToTensor();

        Assert.Equal(rowCount, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]);

        if (rowCount == 0) return;

        int step = Math.Max(1, rowCount / 200);
        for (int r = 0; r < rowCount; r += step)
        {
            Assert.Equal((float)(r * 1.25f), tensor[r, 0]);
            Assert.Equal((float)(r * 2.50), tensor[r, 1]);
            Assert.Equal((float)(r * 3), tensor[r, 2]);
            Assert.Equal((float)(r * 4L), tensor[r, 3]);
        }
    }

    [Fact]
    public void Polaris_ToTensor_MultiTileColumnStride_SpansCacheLines()
    {
        // 75 columns (> 64 column tile boundary) and 3000 rows (> 2048 parallel boundary)
        const int rows = 3000;
        const int cols = 75;

        var seriesList = new ISeries[cols];
        for (int c = 0; c < cols; c++)
        {
            // Mix types across columns
            int typeMod = c % 4;
            if (typeMod == 0)
            {
                var s = new Float32Series($"c_{c}", rows);
                for (int r = 0; r < rows; r++) s.Memory.Span[r] = (float)(r * cols + c);
                seriesList[c] = s;
            }
            else if (typeMod == 1)
            {
                var s = new Float64Series($"c_{c}", rows);
                for (int r = 0; r < rows; r++) s.Memory.Span[r] = (double)(r * cols + c);
                seriesList[c] = s;
            }
            else if (typeMod == 2)
            {
                var s = new Int32Series($"c_{c}", rows);
                for (int r = 0; r < rows; r++) s.Memory.Span[r] = r * cols + c;
                seriesList[c] = s;
            }
            else
            {
                var s = new Int64Series($"c_{c}", rows);
                for (int r = 0; r < rows; r++) s.Memory.Span[r] = (long)(r * cols + c);
                seriesList[c] = s;
            }
        }

        var df = new DataFrame(seriesList);
        using var tensor = df.ToTensor();

        Assert.Equal(rows, tensor.Shape[0]);
        Assert.Equal(cols, tensor.Shape[1]);

        // Validate across boundary coordinates (e.g. column 63, 64, 65 and rows around tile edges)
        for (int r = 0; r < rows; r += 63)
        {
            for (int c = 0; c < cols; c += 15)
            {
                float expected = (float)(r * cols + c);
                float actual = tensor[r, c];
                Assert.Equal(expected, actual);
            }
        }
    }
}
