using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Losses;

/// <summary>
/// Online streaming Fused Cross-Entropy Loss over large vocabularies in pure FP32.
/// Computes online Log-Sum-Exp via Barron's <c>Log1p</c> argmax-excluded split sum and
/// directly backpropagates exact unbiased gradients into hidden states (<c>dFinalHidden</c>)
/// without materializing full <c>[T x Vocab]</c> logit tensors.
/// </summary>
public static unsafe class FusedCrossEntropyLoss
{
    public const int VocabTileSize = 4096;

    /// <summary>
    /// Computes forward-only online streaming Cross-Entropy loss for a dense FP32 LM head weight matrix
    /// <c>[vocabSize, hiddenDim]</c> and hidden activations <c>[seqLen, hiddenDim]</c>.
    /// </summary>
    public static float ComputeLoss(
        float* finalHidden,
        ReadOnlySpan<int> targetTokens,
        float* lmHeadWeight,
        int seqLen,
        int hiddenDim,
        int vocabSize)
        => ComputeLoss(finalHidden, targetTokens, lmHeadWeight, seqLen, hiddenDim, vocabSize, out _);

    /// <summary>
    /// Computes forward-only online streaming Cross-Entropy loss and returns supervised token count.
    /// </summary>
    public static float ComputeLoss(
        float* finalHidden,
        ReadOnlySpan<int> targetTokens,
        float* lmHeadWeight,
        int seqLen,
        int hiddenDim,
        int vocabSize,
        out int validTokenCount)
    {
        ArgumentNullException.ThrowIfNull(finalHidden);
        ArgumentNullException.ThrowIfNull(lmHeadWeight);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        int[] validSeqIndices = new int[seqLen];
        int validCount = 0;
        for (int t = 0; t < seqLen; t++)
        {
            if (t < targetTokens.Length && targetTokens[t] >= 0 && targetTokens[t] < vocabSize)
            {
                validSeqIndices[validCount++] = t;
            }
        }

        validTokenCount = validCount;
        if (validCount == 0) return 0.0f;

        float invValid = 1.0f / validCount;
        nuint validHiddenBytes = checked((nuint)validCount * (nuint)hiddenDim * sizeof(float));
        float* pValidHidden = (float*)NativeMemory.AlignedAlloc(validHiddenBytes, 64);
        int* validTargetIds = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));

        try
        {
            nuint rowCopyBytes = checked((nuint)hiddenDim * sizeof(float));
            for (int i = 0; i < validCount; i++)
            {
                int t = validSeqIndices[i];
                validTargetIds[i] = targetTokens[t];
                Buffer.MemoryCopy(
                    finalHidden + (long)t * hiddenDim,
                    pValidHidden + (long)i * hiddenDim,
                    rowCopyBytes,
                    rowCopyBytes);
            }

            int actualTileSize = Math.Min(VocabTileSize, vocabSize);
            int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;

            float* pTileLogits = (float*)NativeMemory.AlignedAlloc(checked((nuint)validCount * (nuint)actualTileSize * sizeof(float)), 64);
            float* runningMax = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
            int* runningArgMax = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));
            double* runningSumOther = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));
            float* targetLogits = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));

            try
            {
                for (int i = 0; i < validCount; i++)
                {
                    runningMax[i] = float.NegativeInfinity;
                    runningArgMax[i] = -1;
                    runningSumOther[i] = 0.0;
                    targetLogits[i] = 0.0f;
                }

                using var hiddenBlock = new NativeMemoryBlock<float>(pValidHidden, (long)validCount * hiddenDim, ownsMemory: false);
                using var hiddenTensor = new Tensor<float>(hiddenBlock, Shape8.Create(validCount, hiddenDim), Strides8.ComputeContiguous(Shape8.Create(validCount, hiddenDim)), 0, addRef: false);

                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    float* tileWeight = lmHeadWeight + (long)vStart * hiddenDim;

                    using var wBlock = new NativeMemoryBlock<float>(tileWeight, (long)vCount * hiddenDim, ownsMemory: false);
                    using var wTensor = new Tensor<float>(wBlock, Shape8.Create(vCount, hiddenDim), Strides8.ComputeContiguous(Shape8.Create(vCount, hiddenDim)), 0, addRef: false);
                    using var outBlock = new NativeMemoryBlock<float>(pTileLogits, (long)validCount * vCount, ownsMemory: false);
                    using var outTensor = new Tensor<float>(outBlock, Shape8.Create(validCount, vCount), Strides8.ComputeContiguous(Shape8.Create(validCount, vCount)), 0, addRef: false);

                    GpuAccelerator.AcceleratedMatMul(hiddenTensor, wTensor, outTensor, transposeA: false, transposeB: true, target: GpuTarget.Auto);

                    for (int i = 0; i < validCount; i++)
                    {
                        float* logits = pTileLogits + (long)i * vCount;
                        int targetId = validTargetIds[i];
                        if (targetId >= vStart && targetId < vStart + vCount)
                        {
                            targetLogits[i] = logits[targetId - vStart];
                        }

                        UpdateOnlineTileStats(logits, vStart, vCount, ref runningMax[i], ref runningArgMax[i], ref runningSumOther[i]);
                    }
                }

                double totalLoss = 0.0;
                for (int i = 0; i < validCount; i++)
                {
                    totalLoss += Log1p(runningSumOther[i]) + ((double)runningMax[i] - (double)targetLogits[i]);
                }

                return (float)(totalLoss * invValid);
            }
            finally
            {
                NativeMemory.AlignedFree(pTileLogits);
                NativeMemory.Free(runningMax);
                NativeMemory.Free(runningArgMax);
                NativeMemory.Free(runningSumOther);
                NativeMemory.Free(targetLogits);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pValidHidden);
            NativeMemory.Free(validTargetIds);
        }
    }

    /// <summary>
    /// Computes online streaming Cross-Entropy loss and backpropagates exact analytical gradients into <paramref name="dFinalHidden"/>.
    /// </summary>
    public static float ComputeLossAndGradients(
        float* finalHidden,
        ReadOnlySpan<int> targetTokens,
        float* lmHeadWeight,
        float* dFinalHidden,
        int seqLen,
        int hiddenDim,
        int vocabSize)
    {
        ArgumentNullException.ThrowIfNull(finalHidden);
        ArgumentNullException.ThrowIfNull(lmHeadWeight);
        ArgumentNullException.ThrowIfNull(dFinalHidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        long totalHiddenElems = checked((long)seqLen * hiddenDim);
        NativeMemory.Clear(dFinalHidden, checked((nuint)totalHiddenElems * sizeof(float)));

        int[] validSeqIndices = new int[seqLen];
        int validCount = 0;
        for (int t = 0; t < seqLen; t++)
        {
            if (t < targetTokens.Length && targetTokens[t] >= 0 && targetTokens[t] < vocabSize)
            {
                validSeqIndices[validCount++] = t;
            }
        }

        if (validCount == 0) return 0.0f;

        float invValid = 1.0f / validCount;
        nuint validHiddenBytes = checked((nuint)validCount * (nuint)hiddenDim * sizeof(float));
        float* pValidHidden = (float*)NativeMemory.AlignedAlloc(validHiddenBytes, 64);
        float* pValidGrad = (float*)NativeMemory.AlignedAlloc(validHiddenBytes, 64);
        int* validTargetIds = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));

        try
        {
            NativeMemory.Clear(pValidGrad, validHiddenBytes);
            nuint rowCopyBytes = checked((nuint)hiddenDim * sizeof(float));
            for (int i = 0; i < validCount; i++)
            {
                int t = validSeqIndices[i];
                validTargetIds[i] = targetTokens[t];
                Buffer.MemoryCopy(
                    finalHidden + (long)t * hiddenDim,
                    pValidHidden + (long)i * hiddenDim,
                    rowCopyBytes,
                    rowCopyBytes);
            }

            int actualTileSize = Math.Min(VocabTileSize, vocabSize);
            int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;

            float* pTileLogits = (float*)NativeMemory.AlignedAlloc(checked((nuint)validCount * (nuint)actualTileSize * sizeof(float)), 64);
            float* runningMax = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
            int* runningArgMax = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));
            double* runningSumOther = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));
            float* targetLogits = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
            double* invDenom = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));

            try
            {
                for (int i = 0; i < validCount; i++)
                {
                    runningMax[i] = float.NegativeInfinity;
                    runningArgMax[i] = -1;
                    runningSumOther[i] = 0.0;
                    targetLogits[i] = 0.0f;
                }

                using var hiddenBlock = new NativeMemoryBlock<float>(pValidHidden, (long)validCount * hiddenDim, ownsMemory: false);
                using var hiddenTensor = new Tensor<float>(hiddenBlock, Shape8.Create(validCount, hiddenDim), Strides8.ComputeContiguous(Shape8.Create(validCount, hiddenDim)), 0, addRef: false);

                // PASS 1: Online Log-Sum-Exp using Barron's Log1p argmax-excluded split sum
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    float* tileWeight = lmHeadWeight + (long)vStart * hiddenDim;

                    using var wBlock = new NativeMemoryBlock<float>(tileWeight, (long)vCount * hiddenDim, ownsMemory: false);
                    using var wTensor = new Tensor<float>(wBlock, Shape8.Create(vCount, hiddenDim), Strides8.ComputeContiguous(Shape8.Create(vCount, hiddenDim)), 0, addRef: false);
                    using var outBlock = new NativeMemoryBlock<float>(pTileLogits, (long)validCount * vCount, ownsMemory: false);
                    using var outTensor = new Tensor<float>(outBlock, Shape8.Create(validCount, vCount), Strides8.ComputeContiguous(Shape8.Create(validCount, vCount)), 0, addRef: false);

                    GpuAccelerator.AcceleratedMatMul(hiddenTensor, wTensor, outTensor, transposeA: false, transposeB: true, target: GpuTarget.Auto);

                    for (int i = 0; i < validCount; i++)
                    {
                        float* logits = pTileLogits + (long)i * vCount;
                        int targetId = validTargetIds[i];
                        if (targetId >= vStart && targetId < vStart + vCount)
                        {
                            targetLogits[i] = logits[targetId - vStart];
                        }

                        UpdateOnlineTileStats(logits, vStart, vCount, ref runningMax[i], ref runningArgMax[i], ref runningSumOther[i]);
                    }
                }

                double totalLoss = 0.0;
                for (int i = 0; i < validCount; i++)
                {
                    invDenom[i] = 1.0 / (1.0 + runningSumOther[i]);
                    totalLoss += Log1p(runningSumOther[i]) + ((double)runningMax[i] - (double)targetLogits[i]);
                }

                // PASS 2: Tiled analytical softmax gradient + BaseBackward accumulation
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    float* tileWeight = lmHeadWeight + (long)vStart * hiddenDim;

                    if (numTiles > 1)
                    {
                        using var wBlock = new NativeMemoryBlock<float>(tileWeight, (long)vCount * hiddenDim, ownsMemory: false);
                        using var wTensor = new Tensor<float>(wBlock, Shape8.Create(vCount, hiddenDim), Strides8.ComputeContiguous(Shape8.Create(vCount, hiddenDim)), 0, addRef: false);
                        using var outBlock = new NativeMemoryBlock<float>(pTileLogits, (long)validCount * vCount, ownsMemory: false);
                        using var outTensor = new Tensor<float>(outBlock, Shape8.Create(validCount, vCount), Strides8.ComputeContiguous(Shape8.Create(validCount, vCount)), 0, addRef: false);

                        GpuAccelerator.AcceleratedMatMul(hiddenTensor, wTensor, outTensor, transposeA: false, transposeB: true, target: GpuTarget.Auto);
                    }

                    int vStartSnap = vStart;
                    int vCountSnap = vCount;
                    Parallel.For(0, validCount, i =>
                    {
                        WriteTiledSoftmaxCrossEntropyGradients(
                            pTileLogits + (long)i * vCountSnap,
                            vStartSnap,
                            vCountSnap,
                            validTargetIds[i],
                            runningMax[i],
                            runningArgMax[i],
                            runningSumOther[i],
                            invDenom[i],
                            invValid);
                    });

                    TensorBackwardKernels.AccumulateBaseBackward(
                        tileWeight,
                        pTileLogits,
                        pValidGrad,
                        validCount,
                        hiddenDim,
                        vCount);
                }

                for (int i = 0; i < validCount; i++)
                {
                    int t = validSeqIndices[i];
                    Buffer.MemoryCopy(
                        pValidGrad + (long)i * hiddenDim,
                        dFinalHidden + (long)t * hiddenDim,
                        rowCopyBytes,
                        rowCopyBytes);
                }

                return (float)(totalLoss * invValid);
            }
            finally
            {
                NativeMemory.AlignedFree(pTileLogits);
                NativeMemory.Free(runningMax);
                NativeMemory.Free(runningArgMax);
                NativeMemory.Free(runningSumOther);
                NativeMemory.Free(targetLogits);
                NativeMemory.Free(invDenom);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pValidHidden);
            NativeMemory.AlignedFree(pValidGrad);
            NativeMemory.Free(validTargetIds);
        }
    }

    public static float ComputeLoss(Tensor<float> finalHidden, ReadOnlySpan<int> targetTokens, Tensor<float> lmHeadWeight)
    {
        ArgumentNullException.ThrowIfNull(finalHidden);
        ArgumentNullException.ThrowIfNull(lmHeadWeight);
        if (finalHidden.Rank != 2 || lmHeadWeight.Rank != 2)
            throw new ArgumentException("FusedCrossEntropyLoss expects 2D tensors [seqLen, hiddenDim] and [vocabSize, hiddenDim].");
        if (finalHidden.Shape[1] != lmHeadWeight.Shape[1])
            throw new ArgumentException($"Hidden dimension mismatch: {finalHidden.Shape[1]} != {lmHeadWeight.Shape[1]}.");

        using var hC = finalHidden.IsContiguous ? null : finalHidden.Contiguous();
        using var wC = lmHeadWeight.IsContiguous ? null : lmHeadWeight.Contiguous();
        var h = hC ?? finalHidden;
        var w = wC ?? lmHeadWeight;
        return ComputeLoss(h.DataPointer, targetTokens, w.DataPointer, h.Shape[0], h.Shape[1], w.Shape[0]);
    }

    public static float ComputeLossAndGradients(
        Tensor<float> finalHidden,
        ReadOnlySpan<int> targetTokens,
        Tensor<float> lmHeadWeight,
        Tensor<float> dFinalHidden)
    {
        ArgumentNullException.ThrowIfNull(finalHidden);
        ArgumentNullException.ThrowIfNull(lmHeadWeight);
        ArgumentNullException.ThrowIfNull(dFinalHidden);
        if (finalHidden.Rank != 2 || lmHeadWeight.Rank != 2 || dFinalHidden.Rank != 2)
            throw new ArgumentException("FusedCrossEntropyLoss expects 2D tensors.");
        if (finalHidden.Shape != dFinalHidden.Shape)
            throw new ArgumentException("dFinalHidden shape must match finalHidden.");
        if (finalHidden.Shape[1] != lmHeadWeight.Shape[1])
            throw new ArgumentException($"Hidden dimension mismatch: {finalHidden.Shape[1]} != {lmHeadWeight.Shape[1]}.");

        using var hC = finalHidden.IsContiguous ? null : finalHidden.Contiguous();
        using var wC = lmHeadWeight.IsContiguous ? null : lmHeadWeight.Contiguous();
        using var gC = dFinalHidden.IsContiguous ? null : new Tensor<float>(dFinalHidden.Shape);
        var h = hC ?? finalHidden;
        var w = wC ?? lmHeadWeight;
        var g = gC ?? dFinalHidden;

        float loss = ComputeLossAndGradients(h.DataPointer, targetTokens, w.DataPointer, g.DataPointer, h.Shape[0], h.Shape[1], w.Shape[0]);
        if (gC != null)
        {
            Tensor<float>.CopyElements(gC, dFinalHidden);
        }
        return loss;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Log1p(double x)
    {
        if (x == 0.0) return 0.0;
        double u = 1.0 + x;
        if (u == 1.0) return x;
        return Math.Log(u) * (x / (u - 1.0));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float VectorizedMax(float* data, int count)
    {
        int i = 0;
        float maxVal = float.NegativeInfinity;

        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var vMax = Vector512.Create(float.NegativeInfinity);
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                vMax = Vector512.Max(vMax, Vector512.Load(data + i));
            }
            for (int k = 0; k < Vector512<float>.Count; k++)
            {
                if (vMax[k] > maxVal) maxVal = vMax[k];
            }
        }
        else if (Vector256.IsHardwareAccelerated && count >= Vector256<float>.Count)
        {
            var vMax = Vector256.Create(float.NegativeInfinity);
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                vMax = Vector256.Max(vMax, Vector256.Load(data + i));
            }
            for (int k = 0; k < Vector256<float>.Count; k++)
            {
                if (vMax[k] > maxVal) maxVal = vMax[k];
            }
        }
        else if (Vector128.IsHardwareAccelerated && count >= Vector128<float>.Count)
        {
            var vMax = Vector128.Create(float.NegativeInfinity);
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                vMax = Vector128.Max(vMax, Vector128.Load(data + i));
            }
            for (int k = 0; k < Vector128<float>.Count; k++)
            {
                if (vMax[k] > maxVal) maxVal = vMax[k];
            }
        }

        for (; i < count; i++)
        {
            if (data[i] > maxVal) maxVal = data[i];
        }
        return maxVal;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static double VectorizedSumExp(float* data, int count, float maxVal)
    {
        if (float.IsNegativeInfinity(maxVal) || count <= 0)
            return 0.0;

        int i = 0;
        double sum = 0.0;

        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var vMax = Vector512.Create(maxVal);
            var accLo = Vector512<double>.Zero;
            var accHi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var exp = SimdMath.FastExpNonPositiveAvx512(Vector512.Subtract(Vector512.Load(data + i), vMax));
                var (lo, hi) = Vector512.Widen(exp);
                accLo += lo;
                accHi += hi;
            }
            sum += Vector512.Sum(accLo + accHi);
        }
        else if (Vector256.IsHardwareAccelerated && count >= Vector256<float>.Count)
        {
            var vMax = Vector256.Create(maxVal);
            var accLo = Vector256<double>.Zero;
            var accHi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var exp = SimdMath.FastExpNonPositiveAvx2(Vector256.Subtract(Vector256.Load(data + i), vMax));
                var (lo, hi) = Vector256.Widen(exp);
                accLo += lo;
                accHi += hi;
            }
            sum += Vector256.Sum(accLo + accHi);
        }
        else if (Vector128.IsHardwareAccelerated && count >= Vector128<float>.Count)
        {
            var vMax = Vector128.Create(maxVal);
            var accLo = Vector128<double>.Zero;
            var accHi = Vector128<double>.Zero;
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var exp = SimdMath.FastExpNonPositiveAdvSimd(Vector128.Subtract(Vector128.Load(data + i), vMax));
                var (lo, hi) = Vector128.Widen(exp);
                accLo += lo;
                accHi += hi;
            }
            sum += Vector128.Sum(accLo + accHi);
        }

        for (; i < count; i++)
        {
            sum += Math.Exp((double)data[i] - maxVal);
        }
        return sum;
    }

    public static void UpdateOnlineTileStats(
        float* logits,
        int vStart,
        int vCount,
        ref float runningMax,
        ref int runningArgMax,
        ref double runningSumOther)
    {
        float tileMax = float.NegativeInfinity;
        int tileLocalArgMax = 0;
        for (int v = 0; v < vCount; v++)
        {
            float val = logits[v];
            if (v == 0 || val > tileMax)
            {
                tileMax = val;
                tileLocalArgMax = v;
            }
        }

        if (runningArgMax < 0)
        {
            runningMax = tileMax;
            runningArgMax = vStart + tileLocalArgMax;
            runningSumOther = SumExpExcluding(logits, vCount, tileMax, tileLocalArgMax);
        }
        else if (tileMax > runningMax)
        {
            double factor = float.IsNegativeInfinity(runningMax) ? 0.0 : Math.Exp((double)runningMax - tileMax);
            double tileSumOther = SumExpExcluding(logits, vCount, tileMax, tileLocalArgMax);
            runningSumOther = (1.0 + runningSumOther) * factor + tileSumOther;
            runningMax = tileMax;
            runningArgMax = vStart + tileLocalArgMax;
        }
        else
        {
            runningSumOther += SumExpExcluding(logits, vCount, runningMax, excludeLocalIdx: -1);
        }
    }

    public static double SumExpExcluding(float* data, int count, float maxVal, int excludeLocalIdx)
    {
        if (float.IsNegativeInfinity(maxVal) || count <= 0)
            return 0.0;

        if (excludeLocalIdx < 0 || excludeLocalIdx >= count)
            return VectorizedSumExp(data, count, maxVal);

        double left = excludeLocalIdx > 0 ? VectorizedSumExp(data, excludeLocalIdx, maxVal) : 0.0;
        int rightCount = count - excludeLocalIdx - 1;
        double right = rightCount > 0 ? VectorizedSumExp(data + excludeLocalIdx + 1, rightCount, maxVal) : 0.0;
        return left + right;
    }

    public static void WriteTiledSoftmaxCrossEntropyGradients(
        float* logits,
        int vStart,
        int vCount,
        int targetId,
        float maxLogit,
        int globalArgMax,
        double sumOther,
        double invDenom,
        float invValid)
    {
        float scale = (float)(invDenom * invValid);
        int v = 0;

        if (Vector512.IsHardwareAccelerated && vCount >= Vector512<float>.Count)
        {
            var vMax = Vector512.Create(maxLogit);
            var vScale = Vector512.Create(scale);
            int step = Vector512<float>.Count;
            int limit = vCount - step;
            for (; v <= limit; v += step)
            {
                var exp = SimdMath.FastExpNonPositiveAvx512(Vector512.Subtract(Vector512.Load(logits + v), vMax));
                Vector512.Multiply(exp, vScale).Store(logits + v);
            }
        }
        else if (Vector256.IsHardwareAccelerated && vCount >= Vector256<float>.Count)
        {
            var vMax = Vector256.Create(maxLogit);
            var vScale = Vector256.Create(scale);
            int step = Vector256<float>.Count;
            int limit = vCount - step;
            for (; v <= limit; v += step)
            {
                var exp = SimdMath.FastExpNonPositiveAvx2(Vector256.Subtract(Vector256.Load(logits + v), vMax));
                Vector256.Multiply(exp, vScale).Store(logits + v);
            }
        }
        else if (Vector128.IsHardwareAccelerated && vCount >= Vector128<float>.Count)
        {
            var vMax = Vector128.Create(maxLogit);
            var vScale = Vector128.Create(scale);
            int step = Vector128<float>.Count;
            int limit = vCount - step;
            for (; v <= limit; v += step)
            {
                var exp = SimdMath.FastExpNonPositiveAdvSimd(Vector128.Subtract(Vector128.Load(logits + v), vMax));
                Vector128.Multiply(exp, vScale).Store(logits + v);
            }
        }

        for (; v < vCount; v++)
        {
            logits[v] = (float)(Math.Exp((double)logits[v] - maxLogit) * invDenom * invValid);
        }

        // Exact analytical override at argMax and targetId (cancellation-free at globalArgMax)
        int localArgMax = globalArgMax - vStart;
        if ((uint)localArgMax < (uint)vCount)
        {
            logits[localArgMax] = (globalArgMax == targetId)
                ? (float)(-sumOther * invDenom * invValid)
                : (float)(invDenom * invValid);
        }

        int localTarget = targetId - vStart;
        if ((uint)localTarget < (uint)vCount && localTarget != localArgMax)
        {
            logits[localTarget] -= invValid;
        }
    }
}
