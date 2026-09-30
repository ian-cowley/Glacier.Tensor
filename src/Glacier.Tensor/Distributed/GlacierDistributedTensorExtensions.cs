namespace Glacier.Tensor.Distributed;

using System;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Distrib;
using Glacier.Tensor.Core;

/// <summary>
/// First-party distributed computing extensions connecting Glacier.Tensor with Glacier.Compute.
/// Replaces proprietary NVIDIA NCCL with a pure managed C# .NET 10 ring AllReduce architecture.
/// </summary>
public static class GlacierDistributedTensorExtensions
{
    /// <summary>
    /// Executes a distributed ring AllReduce across an in-memory cluster of rank tensors.
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
        int elemCount = (int)rankTensors[0].ElementCount;

        // Verify shape uniformity
        for (int i = 1; i < worldSize; i++)
        {
            if (rankTensors[i].ElementCount != elemCount)
            {
                throw new ArgumentException("All tensors in cluster AllReduce must have identical element counts.");
            }
        }

        var contexts = DistributedAllReduce.CreateLocalCluster(worldSize);
        var buffers = new Memory<float>[worldSize];

        for (int i = 0; i < worldSize; i++)
        {
            buffers[i] = rankTensors[i].ToArray();
        }

        await DistributedAllReduce.ExecuteClusterAsync(contexts, buffers, op, ct);

        for (int i = 0; i < worldSize; i++)
        {
            buffers[i].Span.CopyTo(rankTensors[i].AsSpan());
        }
    }

    /// <summary>
    /// Executes an in-place AllReduce operation for a single tensor against its distributed context.
    /// </summary>
    public static async ValueTask AllReduceAsync(
        this Tensor<float> tensor,
        IDistributedContext context,
        ReductionOp op = ReductionOp.Sum,
        CancellationToken ct = default)
    {
        float[] arr = tensor.ToArray();
        await context.AllReduceAsync(arr, op, ct);
        arr.AsSpan().CopyTo(tensor.AsSpan());
    }
}
