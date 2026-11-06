using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Vectorized SIMD Element-wise Arithmetic and Neural Activation Kernels.
/// </summary>
public static unsafe class ElementwiseKernels
{
    public static void Add(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        ExecuteBinary(a, b, c, (va, vb) => va + vb, (x, y) => x + y);
    }

    public static void Subtract(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        ExecuteBinary(a, b, c, (va, vb) => va - vb, (x, y) => x - y);
    }

    public static void Multiply(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        ExecuteBinary(a, b, c, (va, vb) => va * vb, (x, y) => x * y);
    }

    public static void Divide(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        ExecuteBinary(a, b, c, (va, vb) => va / vb, (x, y) => x / y);
    }

    public static void Scale(Tensor<float> a, float scalar, Tensor<float> c)
    {
        using var aContig = a.Contiguous();
        using var cContig = c.Contiguous();
        var spanA = aContig.AsSpan();
        var spanC = cContig.AsSpan();
        int n = spanA.Length;

        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vScalar = Vector512.Create(scalar);
            for (; i + Vector512<float>.Count <= n; i += Vector512<float>.Count)
            {
                var v = Vector512.LoadUnsafe(ref spanA[i]);
                (v * vScalar).StoreUnsafe(ref spanC[i]);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vScalar = Vector256.Create(scalar);
            for (; i + Vector256<float>.Count <= n; i += Vector256<float>.Count)
            {
                var v = Vector256.LoadUnsafe(ref spanA[i]);
                (v * vScalar).StoreUnsafe(ref spanC[i]);
            }
        }

        for (; i < n; i++)
        {
            spanC[i] = spanA[i] * scalar;
        }

        if (!ReferenceEquals(c, cContig))
        {
            cContig.AsSpan().CopyTo(c.AsSpan());
        }
    }

    public static void ReLU(Tensor<float> input, Tensor<float> output)
    {
        using var inC = input.Contiguous();
        using var outC = output.Contiguous();
        var src = inC.AsSpan();
        var dst = outC.AsSpan();
        int n = src.Length;

        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vZero = Vector512<float>.Zero;
            for (; i + Vector512<float>.Count <= n; i += Vector512<float>.Count)
            {
                var v = Vector512.LoadUnsafe(ref src[i]);
                Vector512.Max(v, vZero).StoreUnsafe(ref dst[i]);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vZero = Vector256<float>.Zero;
            for (; i + Vector256<float>.Count <= n; i += Vector256<float>.Count)
            {
                var v = Vector256.LoadUnsafe(ref src[i]);
                Vector256.Max(v, vZero).StoreUnsafe(ref dst[i]);
            }
        }

        for (; i < n; i++)
        {
            dst[i] = Math.Max(0.0f, src[i]);
        }

        if (!ReferenceEquals(output, outC))
        {
            outC.AsSpan().CopyTo(output.AsSpan());
        }
    }

    public static void ReLUBackward(Tensor<float> gradOutput, Tensor<float> input, Tensor<float> gradInput)
    {
        using var goC = gradOutput.Contiguous();
        using var inC = input.Contiguous();
        using var giC = gradInput.Contiguous();
        var sGo = goC.AsSpan();
        var sIn = inC.AsSpan();
        var sGi = giC.AsSpan();
        int n = sGo.Length;

        for (int i = 0; i < n; i++)
        {
            sGi[i] = sIn[i] > 0.0f ? sGo[i] : 0.0f;
        }

        if (!ReferenceEquals(gradInput, giC))
        {
            giC.AsSpan().CopyTo(gradInput.AsSpan());
        }
    }

    public static void Sigmoid(Tensor<float> input, Tensor<float> output)
    {
        using var inC = input.Contiguous();
        using var outC = output.Contiguous();
        var src = inC.AsSpan();
        var dst = outC.AsSpan();
        int n = src.Length;

        for (int i = 0; i < n; i++)
        {
            dst[i] = 1.0f / (1.0f + MathF.Exp(-src[i]));
        }

        if (!ReferenceEquals(output, outC))
        {
            outC.AsSpan().CopyTo(output.AsSpan());
        }
    }

    public static void Softmax(Tensor<float> input, Tensor<float> output, int dim = -1)
    {
        if (dim < 0) dim += input.Rank;
        if ((uint)dim >= (uint)input.Rank)
            throw new ArgumentOutOfRangeException(nameof(dim));

        using var inC = input.Contiguous();
        using var outC = output.Contiguous();

        int rows = 1;
        for (int i = 0; i < dim; i++) rows *= inC.Shape[i];
        int cols = inC.Shape[dim];
        int inner = 1;
        for (int i = dim + 1; i < inC.Rank; i++) inner *= inC.Shape[i];

        float* pIn = inC.DataPointer;
        float* pOut = outC.DataPointer;

        if (inner == 1)
        {
            for (int r = 0; r < rows; r++)
            {
                float* rowIn = pIn + (long)r * cols;
                float* rowOut = pOut + (long)r * cols;

                float maxVal = rowIn[0];
                for (int c = 1; c < cols; c++)
                {
                    if (rowIn[c] > maxVal) maxVal = rowIn[c];
                }

                float sumExp = 0.0f;
                for (int c = 0; c < cols; c++)
                {
                    float e = MathF.Exp(rowIn[c] - maxVal);
                    rowOut[c] = e;
                    sumExp += e;
                }

                float invSum = 1.0f / sumExp;
                for (int c = 0; c < cols; c++)
                {
                    rowOut[c] *= invSum;
                }
            }
        }
        else
        {
            long outerStride = (long)cols * inner;
            for (int r = 0; r < rows; r++)
            {
                long baseOffset = r * outerStride;
                for (int k = 0; k < inner; k++)
                {
                    long offset = baseOffset + k;
                    float maxVal = pIn[offset];
                    for (int c = 1; c < cols; c++)
                    {
                        float val = pIn[offset + (long)c * inner];
                        if (val > maxVal) maxVal = val;
                    }

                    float sumExp = 0.0f;
                    for (int c = 0; c < cols; c++)
                    {
                        long idx = offset + (long)c * inner;
                        float e = MathF.Exp(pIn[idx] - maxVal);
                        pOut[idx] = e;
                        sumExp += e;
                    }

                    float invSum = 1.0f / sumExp;
                    for (int c = 0; c < cols; c++)
                    {
                        pOut[offset + (long)c * inner] *= invSum;
                    }
                }
            }
        }

        if (!ReferenceEquals(output, outC))
        {
            outC.AsSpan().CopyTo(output.AsSpan());
        }
    }

    public static void SoftmaxBackward(
        Tensor<float> gradOutput,
        Tensor<float> softmaxOutput,
        Tensor<float> gradInput,
        int dim = -1)
    {
        if (dim < 0) dim += softmaxOutput.Rank;
        if ((uint)dim >= (uint)softmaxOutput.Rank)
            throw new ArgumentOutOfRangeException(nameof(dim));

        using var goC = gradOutput.Contiguous();
        using var yC = softmaxOutput.Contiguous();
        using var giC = gradInput.Contiguous();

        int rows = 1;
        for (int i = 0; i < dim; i++) rows *= yC.Shape[i];
        int cols = yC.Shape[dim];
        int inner = 1;
        for (int i = dim + 1; i < yC.Rank; i++) inner *= yC.Shape[i];

        float* pGo = goC.DataPointer;
        float* pY = yC.DataPointer;
        float* pGi = giC.DataPointer;

        if (inner == 1)
        {
            for (int r = 0; r < rows; r++)
            {
                long rowBase = (long)r * cols;
                float dot = 0.0f;
                for (int c = 0; c < cols; c++)
                {
                    dot += pGo[rowBase + c] * pY[rowBase + c];
                }

                for (int c = 0; c < cols; c++)
                {
                    long idx = rowBase + c;
                    pGi[idx] = pY[idx] * (pGo[idx] - dot);
                }
            }
        }
        else
        {
            long outerStride = (long)cols * inner;
            for (int r = 0; r < rows; r++)
            {
                long baseOffset = r * outerStride;
                for (int k = 0; k < inner; k++)
                {
                    long offset = baseOffset + k;
                    float dot = 0.0f;
                    for (int c = 0; c < cols; c++)
                    {
                        long idx = offset + (long)c * inner;
                        dot += pGo[idx] * pY[idx];
                    }

                    for (int c = 0; c < cols; c++)
                    {
                        long idx = offset + (long)c * inner;
                        pGi[idx] = pY[idx] * (pGo[idx] - dot);
                    }
                }
            }
        }

        if (!ReferenceEquals(gradInput, giC))
        {
            giC.AsSpan().CopyTo(gradInput.AsSpan());
        }
    }

    public static void LayerNorm(
        Tensor<float> input,
        Tensor<float>? gamma,
        Tensor<float>? beta,
        Tensor<float> output,
        float eps = 1e-5f)
    {
        using var inC = input.Contiguous();
        using var outC = output.Contiguous();
        using var gC = gamma?.Contiguous();
        using var bC = beta?.Contiguous();

        int d = inC.Shape[inC.Rank - 1];
        int batch = (int)(inC.ElementCount / d);

        var inSpan = inC.AsSpan();
        var outSpan = outC.AsSpan();
        Span<float> gSpan = gC != null ? gC.AsSpan() : default;
        Span<float> bSpan = bC != null ? bC.AsSpan() : default;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * d;

            float mean = 0f;
            for (int i = 0; i < d; i++) mean += inSpan[offset + i];
            mean /= d;

            float var = 0f;
            for (int i = 0; i < d; i++)
            {
                float diff = inSpan[offset + i] - mean;
                var += diff * diff;
            }
            var /= d;
            float invStd = 1.0f / MathF.Sqrt(var + eps);

            for (int i = 0; i < d; i++)
            {
                float norm = (inSpan[offset + i] - mean) * invStd;
                float scaled = gamma != null ? norm * gSpan[i] : norm;
                outSpan[offset + i] = beta != null ? scaled + bSpan[i] : scaled;
            }
        }

        if (!ReferenceEquals(output, outC))
        {
            outC.AsSpan().CopyTo(output.AsSpan());
        }
    }

    public static void LayerNormBackward(
        Tensor<float> gradOutput,
        Tensor<float> input,
        Tensor<float>? gamma,
        Tensor<float> gradInput,
        Tensor<float>? gradGamma,
        Tensor<float>? gradBeta,
        float eps = 1e-5f)
    {
        using var goC = gradOutput.Contiguous();
        using var inC = input.Contiguous();
        using var gC = gamma?.Contiguous();
        using var giC = gradInput.Contiguous();
        using var ggC = gradGamma?.Contiguous();
        using var gbC = gradBeta?.Contiguous();

        int d = inC.Shape[inC.Rank - 1];
        int batch = (int)(inC.ElementCount / d);

        var goSpan = goC.AsSpan();
        var inSpan = inC.AsSpan();
        Span<float> gSpan = gC != null ? gC.AsSpan() : default;
        var giSpan = giC.AsSpan();
        Span<float> ggSpan = ggC != null ? ggC.AsSpan() : default;
        Span<float> gbSpan = gbC != null ? gbC.AsSpan() : default;

        if (gradGamma != null) ggSpan.Clear();
        if (gradBeta != null) gbSpan.Clear();

        float invD = 1.0f / d;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * d;

            float mean = 0f;
            for (int i = 0; i < d; i++) mean += inSpan[offset + i];
            mean *= invD;

            float var = 0f;
            for (int i = 0; i < d; i++)
            {
                float diff = inSpan[offset + i] - mean;
                var += diff * diff;
            }
            var *= invD;
            float invStd = 1.0f / MathF.Sqrt(var + eps);

            float sumDxHat = 0f;
            float sumDxHatXHat = 0f;

            for (int i = 0; i < d; i++)
            {
                float dy = goSpan[offset + i];
                float xHat = (inSpan[offset + i] - mean) * invStd;
                float g = gamma != null ? dy * gSpan[i] : dy;

                sumDxHat += g;
                sumDxHatXHat += g * xHat;

                if (gradGamma != null) ggSpan[i] += dy * xHat;
                if (gradBeta != null) gbSpan[i] += dy;
            }

            for (int i = 0; i < d; i++)
            {
                float dy = goSpan[offset + i];
                float xHat = (inSpan[offset + i] - mean) * invStd;
                float g = gamma != null ? dy * gSpan[i] : dy;
                giSpan[offset + i] = invStd * (g - invD * sumDxHat - invD * xHat * sumDxHatXHat);
            }
        }

        if (!ReferenceEquals(gradInput, giC))
        {
            giC.AsSpan().CopyTo(gradInput.AsSpan());
        }
        if (gradGamma != null && !ReferenceEquals(gradGamma, ggC))
        {
            ggC!.AsSpan().CopyTo(gradGamma.AsSpan());
        }
        if (gradBeta != null && !ReferenceEquals(gradBeta, gbC))
        {
            gbC!.AsSpan().CopyTo(gradBeta.AsSpan());
        }
    }

    public static void GELU(Tensor<float> input, Tensor<float> output)
    {
        using var inC = input.Contiguous();
        using var outC = output.Contiguous();
        var src = inC.AsSpan();
        var dst = outC.AsSpan();
        int n = src.Length;

        const float sqrt2OverPi = 0.7978845608f;
        const float coeff = 0.044715f;

        for (int i = 0; i < n; i++)
        {
            float x = src[i];
            float u = sqrt2OverPi * (x + coeff * x * x * x);
            float t = MathF.Tanh(u);
            dst[i] = 0.5f * x * (1.0f + t);
        }

        if (!ReferenceEquals(output, outC))
        {
            outC.AsSpan().CopyTo(output.AsSpan());
        }
    }

    public static void GELUBackward(Tensor<float> gradOutput, Tensor<float> input, Tensor<float> gradInput)
    {
        using var goC = gradOutput.Contiguous();
        using var inC = input.Contiguous();
        using var giC = gradInput.Contiguous();
        var sGo = goC.AsSpan();
        var sIn = inC.AsSpan();
        var sGi = giC.AsSpan();
        int n = sGo.Length;

        const float sqrt2OverPi = 0.7978845608f;
        const float coeff = 0.044715f;
        const float coeff3 = 3.0f * coeff;

        for (int i = 0; i < n; i++)
        {
            float x = sIn[i];
            float x2 = x * x;
            float u = sqrt2OverPi * (x + coeff * x * x2);
            float t = MathF.Tanh(u);
            float oneMinusT2 = 1.0f - t * t;
            float duDx = sqrt2OverPi * (1.0f + coeff3 * x2);
            float dGelu = 0.5f * (1.0f + t) + 0.5f * x * oneMinusT2 * duDx;
            sGi[i] = sGo[i] * dGelu;
        }

        if (!ReferenceEquals(gradInput, giC))
        {
            giC.AsSpan().CopyTo(gradInput.AsSpan());
        }
    }

    public static void RMSNorm(Tensor<float> input, Tensor<float>? weight, Tensor<float> output, float eps = 1e-5f)
    {
        using var inC = input.Contiguous();
        using var outC = output.Contiguous();
        using var wC = weight?.Contiguous();

        int d = inC.Shape[inC.Rank - 1];
        int batch = (int)(inC.ElementCount / d);

        var inSpan = inC.AsSpan();
        var outSpan = outC.AsSpan();
        Span<float> wSpan = wC != null ? wC.AsSpan() : default;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * d;
            float sumSq = 0.0f;
            for (int i = 0; i < d; i++)
            {
                float val = inSpan[offset + i];
                sumSq += val * val;
            }
            float rms = MathF.Sqrt(sumSq / d + eps);
            float invRms = 1.0f / rms;

            if (weight != null)
            {
                for (int i = 0; i < d; i++)
                {
                    outSpan[offset + i] = inSpan[offset + i] * invRms * wSpan[i];
                }
            }
            else
            {
                for (int i = 0; i < d; i++)
                {
                    outSpan[offset + i] = inSpan[offset + i] * invRms;
                }
            }
        }

        if (!ReferenceEquals(output, outC))
        {
            outC.AsSpan().CopyTo(output.AsSpan());
        }
    }

    public static void RMSNormBackward(
        Tensor<float> gradOutput,
        Tensor<float> input,
        Tensor<float>? weight,
        Tensor<float> gradInput,
        Tensor<float>? gradWeight,
        float eps = 1e-5f)
    {
        using var goC = gradOutput.Contiguous();
        using var inC = input.Contiguous();
        using var wC = weight?.Contiguous();
        using var giC = gradInput.Contiguous();
        using var gwC = gradWeight?.Contiguous();

        int d = inC.Shape[inC.Rank - 1];
        int batch = (int)(inC.ElementCount / d);

        var goSpan = goC.AsSpan();
        var inSpan = inC.AsSpan();
        Span<float> wSpan = wC != null ? wC.AsSpan() : default;
        var giSpan = giC.AsSpan();
        Span<float> gwSpan = gwC != null ? gwC.AsSpan() : default;

        if (gradWeight != null)
        {
            gwSpan.Clear();
        }

        for (int b = 0; b < batch; b++)
        {
            int offset = b * d;

            float sumSq = 0.0f;
            for (int i = 0; i < d; i++)
            {
                float val = inSpan[offset + i];
                sumSq += val * val;
            }
            float rms = MathF.Sqrt(sumSq / d + eps);
            float invRms = 1.0f / rms;

            float dotGAndXHat = 0.0f;
            for (int i = 0; i < d; i++)
            {
                float dy = goSpan[offset + i];
                float xHat = inSpan[offset + i] * invRms;
                float w = (weight != null) ? wSpan[i] : 1.0f;
                float g = dy * w;
                dotGAndXHat += g * xHat;

                if (gradWeight != null)
                {
                    gwSpan[i] += dy * xHat;
                }
            }

            for (int i = 0; i < d; i++)
            {
                float dy = goSpan[offset + i];
                float xHat = inSpan[offset + i] * invRms;
                float w = (weight != null) ? wSpan[i] : 1.0f;
                float g = dy * w;
                giSpan[offset + i] = invRms * (g - (xHat / d) * dotGAndXHat);
            }
        }

        if (!ReferenceEquals(gradInput, giC))
        {
            giC.AsSpan().CopyTo(gradInput.AsSpan());
        }
        if (gradWeight != null && !ReferenceEquals(gradWeight, gwC))
        {
            gwC!.AsSpan().CopyTo(gradWeight.AsSpan());
        }
    }

    private static void ExecuteBinary(
        Tensor<float> a, Tensor<float> b, Tensor<float> c,
        Func<Vector256<float>, Vector256<float>, Vector256<float>> vecOp,
        Func<float, float, float> scalarOp)
    {
        if (a.ElementCount != b.ElementCount || a.ElementCount != c.ElementCount)
            throw new ArgumentException("Tensor element counts must match for elementwise operations.");

        using var aContig = a.Contiguous();
        using var bContig = b.Contiguous();
        using var cContig = c.Contiguous();

        var spanA = aContig.AsSpan();
        var spanB = bContig.AsSpan();
        var spanC = cContig.AsSpan();
        int n = spanA.Length;

        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i + Vector256<float>.Count <= n; i += Vector256<float>.Count)
            {
                var va = Vector256.LoadUnsafe(ref spanA[i]);
                var vb = Vector256.LoadUnsafe(ref spanB[i]);
                vecOp(va, vb).StoreUnsafe(ref spanC[i]);
            }
        }

        for (; i < n; i++)
        {
            spanC[i] = scalarOp(spanA[i], spanB[i]);
        }

        if (!ReferenceEquals(c, cContig))
        {
            cContig.AsSpan().CopyTo(c.AsSpan());
        }
    }
}
