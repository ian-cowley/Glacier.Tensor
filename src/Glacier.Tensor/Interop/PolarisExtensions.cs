using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Interop;

/// <summary>
/// Zero-copy data bridge between Glacier.Polaris columnar DataFrames and Glacier.Tensor.
/// </summary>
public static unsafe class PolarisExtensions
{
    private enum ColumnKind : byte
    {
        Float32,
        Float64,
        Int32,
        Int64
    }

    private readonly struct ColumnReader
    {
        public readonly ColumnKind Kind;
        public readonly void* Pointer;

        public ColumnReader(ColumnKind kind, void* pointer)
        {
            Kind = kind;
            Pointer = pointer;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float ReadFloat(int row)
        {
            return Kind switch
            {
                ColumnKind.Float32 => ((float*)Pointer)[row],
                ColumnKind.Float64 => (float)((double*)Pointer)[row],
                ColumnKind.Int32 => (float)((int*)Pointer)[row],
                ColumnKind.Int64 => (float)((long*)Pointer)[row],
                _ => 0f
            };
        }
    }

    /// <summary>
    /// Converts selected numeric columns of a Polaris DataFrame into a 2D Tensor with shape [Rows, Columns].
    /// Employs 2D 64x64 L1 cache-blocked tiling with pre-pinned column memory pointers and multi-threaded
    /// row block execution via Parallel.For when rows >= 2048 to eliminate column-stride cache thrashing.
    /// </summary>
    public static Tensor<float> ToTensor(this DataFrame df, params string[] columnNames)
    {
        if (df == null) throw new ArgumentNullException(nameof(df));

        var cols = (columnNames.Length > 0 ? columnNames : df.Columns.Select(c => c.Name)).ToArray();
        int rows = df.RowCount;
        int numCols = cols.Length;

        var tensor = new Tensor<float>(rows, numCols);
        if (rows == 0 || numCols == 0)
        {
            return tensor;
        }

        // Fast-path: Single column direct memory copy
        if (numCols == 1)
        {
            var singleSeries = df.Columns.FirstOrDefault(x => x.Name.Equals(cols[0], StringComparison.OrdinalIgnoreCase))
                ?? throw new ArgumentException($"Column '{cols[0]}' not found in DataFrame.");

            if (singleSeries is Series<float> sF32)
            {
                sF32.Memory.Span.CopyTo(tensor.AsSpan());
                return tensor;
            }
        }

        // Pre-resolve and pin column memory buffers
        var handles = new MemoryHandle[numCols];
        var readers = new ColumnReader[numCols];
        bool allFloat32 = true;

        try
        {
            for (int c = 0; c < numCols; c++)
            {
                string colName = cols[c];
                var series = df.Columns.FirstOrDefault(x => x.Name.Equals(colName, StringComparison.OrdinalIgnoreCase))
                    ?? throw new ArgumentException($"Column '{colName}' not found in DataFrame.");

                if (series is Series<float> f32)
                {
                    handles[c] = f32.Memory.Pin();
                    readers[c] = new ColumnReader(ColumnKind.Float32, handles[c].Pointer);
                }
                else if (series is Series<double> f64)
                {
                    allFloat32 = false;
                    handles[c] = f64.Memory.Pin();
                    readers[c] = new ColumnReader(ColumnKind.Float64, handles[c].Pointer);
                }
                else if (series is Series<int> i32)
                {
                    allFloat32 = false;
                    handles[c] = i32.Memory.Pin();
                    readers[c] = new ColumnReader(ColumnKind.Int32, handles[c].Pointer);
                }
                else if (series is Series<long> i64)
                {
                    allFloat32 = false;
                    handles[c] = i64.Memory.Pin();
                    readers[c] = new ColumnReader(ColumnKind.Int64, handles[c].Pointer);
                }
                else
                {
                    throw new NotSupportedException($"Series type {series.GetType().Name} is not a supported numeric column for tensor conversion.");
                }
            }

            float* pTensor = tensor.DataPointer;
            const int TileRows = 64;
            const int TileCols = 64;

            if (allFloat32)
            {
                // Specialized high-throughput Float32 fast path
                float*[] colPtrs = new float*[numCols];
                for (int c = 0; c < numCols; c++)
                {
                    colPtrs[c] = (float*)readers[c].Pointer;
                }

                if (rows >= 2048)
                {
                    int totalRowBlocks = (rows + TileRows - 1) / TileRows;
                    Parallel.For(0, totalRowBlocks, blockIdx =>
                    {
                        int rStart = blockIdx * TileRows;
                        int rEnd = Math.Min(rStart + TileRows, rows);

                        for (int cStart = 0; cStart < numCols; cStart += TileCols)
                        {
                            int cEnd = Math.Min(cStart + TileCols, numCols);
                            for (int r = rStart; r < rEnd; r++)
                            {
                                float* pDestRow = pTensor + (long)r * numCols;
                                for (int c = cStart; c < cEnd; c++)
                                {
                                    pDestRow[c] = colPtrs[c][r];
                                }
                            }
                        }
                    });
                }
                else
                {
                    for (int rStart = 0; rStart < rows; rStart += TileRows)
                    {
                        int rEnd = Math.Min(rStart + TileRows, rows);
                        for (int cStart = 0; cStart < numCols; cStart += TileCols)
                        {
                            int cEnd = Math.Min(cStart + TileCols, numCols);
                            for (int r = rStart; r < rEnd; r++)
                            {
                                float* pDestRow = pTensor + (long)r * numCols;
                                for (int c = cStart; c < cEnd; c++)
                                {
                                    pDestRow[c] = colPtrs[c][r];
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // General heterogeneous numeric path with L1 cache-blocked tiles
                if (rows >= 2048)
                {
                    int totalRowBlocks = (rows + TileRows - 1) / TileRows;
                    Parallel.For(0, totalRowBlocks, blockIdx =>
                    {
                        int rStart = blockIdx * TileRows;
                        int rEnd = Math.Min(rStart + TileRows, rows);

                        for (int cStart = 0; cStart < numCols; cStart += TileCols)
                        {
                            int cEnd = Math.Min(cStart + TileCols, numCols);
                            for (int r = rStart; r < rEnd; r++)
                            {
                                float* pDestRow = pTensor + (long)r * numCols;
                                for (int c = cStart; c < cEnd; c++)
                                {
                                    pDestRow[c] = readers[c].ReadFloat(r);
                                }
                            }
                        }
                    });
                }
                else
                {
                    for (int rStart = 0; rStart < rows; rStart += TileRows)
                    {
                        int rEnd = Math.Min(rStart + TileRows, rows);
                        for (int cStart = 0; cStart < numCols; cStart += TileCols)
                        {
                            int cEnd = Math.Min(cStart + TileCols, numCols);
                            for (int r = rStart; r < rEnd; r++)
                            {
                                float* pDestRow = pTensor + (long)r * numCols;
                                for (int c = cStart; c < cEnd; c++)
                                {
                                    pDestRow[c] = readers[c].ReadFloat(r);
                                }
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            for (int c = 0; c < numCols; c++)
            {
                handles[c].Dispose();
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
