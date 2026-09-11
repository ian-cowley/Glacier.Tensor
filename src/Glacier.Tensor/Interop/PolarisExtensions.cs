using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Interop;

/// <summary>
/// Zero-copy data bridge between Glacier.Polaris columnar DataFrames and Glacier.Tensor.
/// </summary>
public static class PolarisExtensions
{
    /// <summary>
    /// Converts selected numeric columns of a Polaris DataFrame into a 2D Tensor with shape [Rows, Columns].
    /// </summary>
    public static Tensor<float> ToTensor(this DataFrame df, params string[] columnNames)
    {
        if (df == null) throw new ArgumentNullException(nameof(df));

        var cols = (columnNames.Length > 0 ? columnNames : df.Columns.Select(c => c.Name)).ToArray();
        int rows = df.RowCount;
        int numCols = cols.Length;

        var tensor = new Tensor<float>(rows, numCols);

        for (int c = 0; c < numCols; c++)
        {
            string colName = cols[c];
            var series = df.Columns.FirstOrDefault(x => x.Name.Equals(colName, StringComparison.OrdinalIgnoreCase))
                ?? throw new ArgumentException($"Column '{colName}' not found in DataFrame.");

            if (series is Series<float> f32)
            {
                var span = f32.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    tensor[r, c] = span[r];
                }
            }
            else if (series is Series<double> f64)
            {
                var span = f64.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    tensor[r, c] = (float)span[r];
                }
            }
            else if (series is Series<int> i32)
            {
                var span = i32.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    tensor[r, c] = (float)span[r];
                }
            }
            else if (series is Series<long> i64)
            {
                var span = i64.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    tensor[r, c] = (float)span[r];
                }
            }
            else
            {
                throw new NotSupportedException($"Series type {series.GetType().Name} is not a supported numeric column for tensor conversion.");
            }
        }

        return tensor;
    }

    /// <summary>
    /// Converts a 1D Tensor into a Polaris Float32Series.
    /// </summary>
    public static Float32Series ToSeries(this Tensor<float> tensor, string name = "tensor_series")
    {
        if (tensor.Rank != 1)
            throw new ArgumentException("ToSeries requires a 1D tensor.");

        using var contig = tensor.Contiguous();
        int len = (int)contig.ElementCount;
        var series = new Float32Series(name, len);
        contig.AsSpan().CopyTo(series.Memory.Span);
        return series;
    }
}
