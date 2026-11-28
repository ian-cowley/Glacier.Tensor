using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Pure FP32 SIMD backward accumulation kernels for dense linear and language-model head backpropagation.
/// </summary>
public static unsafe class TensorBackwardKernels
{
    private const int BaseColBlock = 256;

    /// <summary>
    /// Accumulates FP32 input gradients: <c>dXInput[seqLen, inFeatures] += dY[seqLen, outFeatures] * W_base[outFeatures, inFeatures]</c>.
    /// Uses cache-tiled columns and lock-free SIMD FMA across sequence and feature blocks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AccumulateBaseBackward(
        float* baseWeightData,
        float* pDY,
        float* pDXInput,
        int seqLen,
        int inFeatures,
        int outFeatures)
    {
        if (baseWeightData == null || pDY == null || pDXInput == null || seqLen <= 0 || inFeatures <= 0 || outFeatures <= 0)
        {
            return;
        }

        int colBlocks = (inFeatures + BaseColBlock - 1) / BaseColBlock;
        int totalSlices = seqLen * colBlocks;

        if (totalSlices >= 2 && (long)seqLen * inFeatures * outFeatures >= 16384)
        {
            Parallel.For(0, totalSlices, workIdx =>
            {
                int s = workIdx / colBlocks;
                int colStart = (workIdx % colBlocks) * BaseColBlock;
                int colLen = Math.Min(BaseColBlock, inFeatures - colStart);
                float* pDXSlice = pDXInput + (long)s * inFeatures + colStart;
                float* pDYRow = pDY + (long)s * outFeatures;

                for (int o = 0; o < outFeatures; o++)
                {
                    float dy = pDYRow[o];
                    if (dy == 0f) continue;
                    AccumulateScaledSimd(pDXSlice, baseWeightData + (long)o * inFeatures + colStart, dy, colLen);
                }
            });
        }
        else
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* pDXRow = pDXInput + (long)s * inFeatures;
                float* pDYRow = pDY + (long)s * outFeatures;
                for (int o = 0; o < outFeatures; o++)
                {
                    float dy = pDYRow[o];
                    if (dy == 0f) continue;
                    AccumulateScaledSimd(pDXRow, baseWeightData + (long)o * inFeatures, dy, inFeatures);
                }
            }
        }
    }

    /// <summary>
    /// Vectorized <c>dst[i] += scale * src[i]</c> across AVX-512, AVX2, ARM64/SSE2, and scalar tail.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void AccumulateScaledSimd(float* dst, float* src, float scale, int count)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var vScale = Vector512.Create(scale);
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector512.Load(dst + i);
                var vSrc = Vector512.Load(src + i);
                Vector512.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
            }
        }

        if (Vector256.IsHardwareAccelerated && count - i >= Vector256<float>.Count)
        {
            var vScale = Vector256.Create(scale);
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector256.Load(dst + i);
                var vSrc = Vector256.Load(src + i);
                Vector256.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
            }
        }

        if (Vector128.IsHardwareAccelerated && count - i >= Vector128<float>.Count)
        {
            var vScale = Vector128.Create(scale);
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector128.Load(dst + i);
                var vSrc = Vector128.Load(src + i);
                Vector128.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
            }
        }

        for (; i < count; i++)
        {
            dst[i] += scale * src[i];
        }
    }
}
