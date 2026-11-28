namespace Glacier.Tensor.Distributed;

using System;
using System.Buffers;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Distrib;
using Glacier.Tensor.Core;

/// <summary>
/// Distributed computing extensions connecting Glacier.Tensor with Glacier.Compute ring AllReduce transports.
/// </summary>
public static class GlacierDistributedTensorExtensions
{
    /// <summary>
    /// Executes an in-process local ring AllReduce simulation across <paramref name="rankTensors"/>
    /// backed by <see cref="DistributedAllReduce.CreateLocalCluster"/> (<c>InMemoryRingTransport</c>).
    /// For multi-process / cross-node reduction over sockets, use <see cref="AllReduceAsync"/> with a networked <see cref="IDistributedContext"/>.
    /// </summary>
    public static async ValueTask AllReduceClusterAsync(
        this Tensor<float>[] rankTensors,
        ReductionOp op = ReductionOp.Sum,
        CancellationToken ct = default)
    {
        if (rankTensors == null || rankTensors.Length == 0)
        {
            throw new ArgumentException("Rank tensors array must not be empty.", nameof(rankTensors));
        }

        int worldSize = rankTensors.Length;
        long totalCount = rankTensors[0].ElementCount;
        if (totalCount > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(rankTensors), $"Tensor element count ({totalCount}) exceeds maximum single-buffer AllReduce length ({int.MaxValue}).");
        }
        int elemCount = (int)totalCount;

        // Verify shape uniformity
        for (int i = 1; i < worldSize; i++)
        {
            if (rankTensors[i].ElementCount != totalCount)
            {
                throw new ArgumentException("All tensors in cluster AllReduce must have identical element counts.");
            }
        }

        var contexts = DistributedAllReduce.CreateLocalCluster(worldSize);
        var rentedArrays = new float[worldSize][];
        var buffers = new Memory<float>[worldSize];

        try
        {
            for (int i = 0; i < worldSize; i++)
            {
                float[] rented = ArrayPool<float>.Shared.Rent(elemCount);
                rentedArrays[i] = rented;
                if (rankTensors[i].IsContiguous)
                {
                    rankTensors[i].AsSpan().CopyTo(rented.AsSpan(0, elemCount));
                }
                else
                {
                    using var contig = rankTensors[i].Contiguous();
                    contig.AsSpan().CopyTo(rented.AsSpan(0, elemCount));
                }
                buffers[i] = rented.AsMemory(0, elemCount);
            }

            await DistributedAllReduce.ExecuteClusterAsync(contexts, buffers, op, ct);

            for (int i = 0; i < worldSize; i++)
            {
                if (rankTensors[i].IsContiguous)
                {
                    buffers[i].Span.CopyTo(rankTensors[i].AsSpan());
                }
                else
                {
                    using var tmp = Tensor<float>.FromSpan(buffers[i].Span, rankTensors[i].Shape.AsSpan());
                    Tensor<float>.CopyElements(tmp, rankTensors[i]);
                }
            }
        }
        finally
        {
            for (int i = 0; i < worldSize; i++)
            {
                if (rentedArrays[i] != null)
                {
                    ArrayPool<float>.Shared.Return(rentedArrays[i]);
                }
            }
        }
    }

    /// <summary>
    /// Executes an in-place AllReduce operation for a single tensor against its <see cref="IDistributedContext"/>.
    /// </summary>
    public static async ValueTask AllReduceAsync(
        this Tensor<float> tensor,
        IDistributedContext context,
        ReductionOp op = ReductionOp.Sum,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(context);

        long totalCount = tensor.ElementCount;
        if (totalCount > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(tensor), $"Tensor element count ({totalCount}) exceeds maximum single-buffer AllReduce length ({int.MaxValue}).");
        }
        int elemCount = (int)totalCount;

        float[] rented = ArrayPool<float>.Shared.Rent(elemCount);
        try
        {
            var buf = rented.AsMemory(0, elemCount);
            if (tensor.IsContiguous)
            {
                tensor.AsSpan().CopyTo(buf.Span);
            }
            else
            {
                using var contig = tensor.Contiguous();
                contig.AsSpan().CopyTo(buf.Span);
            }

            await context.AllReduceAsync(buf, op, ct);

            if (tensor.IsContiguous)
            {
                buf.Span.CopyTo(tensor.AsSpan());
            }
            else
            {
                using var tmp = Tensor<float>.FromSpan(buf.Span, tensor.Shape.AsSpan());
                Tensor<float>.CopyElements(tmp, tensor);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }
}
