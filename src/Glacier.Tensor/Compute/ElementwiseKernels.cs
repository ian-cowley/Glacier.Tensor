using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Vectorized SIMD Element-wise Arithmetic and Neural Activation Kernels across AVX-512, AVX2, Vector128, and scalar paths.
/// </summary>
public static unsafe class ElementwiseKernels
{
    private const int ParallelElementThreshold = 32768;

    private interface IBinaryOp
    {
        static abstract Vector512<float> Apply(Vector512<float> a, Vector512<float> b);
        static abstract Vector256<float> Apply(Vector256<float> a, Vector256<float> b);
        static abstract Vector128<float> Apply(Vector128<float> a, Vector128<float> b);
        static abstract float Apply(float a, float b);
    }

    private readonly struct AddOp : IBinaryOp
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector512<float> Apply(Vector512<float> a, Vector512<float> b) => a + b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a + b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a + b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Apply(float a, float b) => a + b;
    }

    private readonly struct SubtractOp : IBinaryOp
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector512<float> Apply(Vector512<float> a, Vector512<float> b) => a - b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a - b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a - b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Apply(float a, float b) => a - b;
    }

    private readonly struct MultiplyOp : IBinaryOp
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector512<float> Apply(Vector512<float> a, Vector512<float> b) => a * b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a * b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a * b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Apply(float a, float b) => a * b;
    }

    private readonly struct DivideOp : IBinaryOp
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector512<float> Apply(Vector512<float> a, Vector512<float> b) => a / b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a / b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a / b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Apply(float a, float b) => a / b;
    }

    public static void Add(Tensor<float> a, Tensor<float> b, Tensor<float> c) => ExecuteBinary<AddOp>(a, b, c);
    public static void Subtract(Tensor<float> a, Tensor<float> b, Tensor<float> c) => ExecuteBinary<SubtractOp>(a, b, c);
    public static void Multiply(Tensor<float> a, Tensor<float> b, Tensor<float> c) => ExecuteBinary<MultiplyOp>(a, b, c);
    public static void Divide(Tensor<float> a, Tensor<float> b, Tensor<float> c) => ExecuteBinary<DivideOp>(a, b, c);

    public static void Scale(Tensor<float> a, float scalar, Tensor<float> c)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(c);
        if (a.ElementCount != c.ElementCount)
            throw new ArgumentException("Tensor element counts must match for scaling.");

        using var aContig = (a.IsContiguous && a.ElementOffset == 0) ? null : a.Contiguous();
        using var cContig = (c.IsContiguous && c.ElementOffset == 0) ? null : new Tensor<float>(c.Shape);
        var aSrc = aContig ?? a;
        var cDst = cContig ?? c;

        int n = checked((int)aSrc.ElementCount);
        float* pA = aSrc.DataPointer;
        float* pC = cDst.DataPointer;

        if (n >= ParallelElementThreshold)
        {
            ParallelChunked(n, (start, len) => ScaleChunk(pA + start, pC + start, scalar, len));
        }
        else
        {
            ScaleChunk(pA, pC, scalar, n);
        }

        if (cContig != null)
        {
            Tensor<float>.CopyElements(cContig, c);
        }
    }

    public static void ReLU(Tensor<float> input, Tensor<float> output)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        if (input.ElementCount != output.ElementCount)
            throw new ArgumentException("Tensor element counts must match.");

        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var outOwned = (output.IsContiguous && output.ElementOffset == 0) ? null : new Tensor<float>(output.Shape);
        var inC = inOwned ?? input;
        var outC = outOwned ?? output;

        int n = checked((int)inC.ElementCount);
        float* pIn = inC.DataPointer;
        float* pOut = outC.DataPointer;

        if (n >= ParallelElementThreshold)
        {
            ParallelChunked(n, (start, len) => ReluChunk(pIn + start, pOut + start, len));
        }
        else
        {
            ReluChunk(pIn, pOut, n);
        }

        if (outOwned != null)
        {
            Tensor<float>.CopyElements(outOwned, output);
        }
    }

    public static void ReLUBackward(Tensor<float> gradOutput, Tensor<float> input, Tensor<float> gradInput)
    {
        ArgumentNullException.ThrowIfNull(gradOutput);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gradInput);
        if (gradOutput.ElementCount != input.ElementCount || input.ElementCount != gradInput.ElementCount)
            throw new ArgumentException("Tensor element counts must match.");

        using var goOwned = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0) ? null : gradOutput.Contiguous();
        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var giOwned = (gradInput.IsContiguous && gradInput.ElementOffset == 0) ? null : new Tensor<float>(gradInput.Shape);
        var goC = goOwned ?? gradOutput;
        var inC = inOwned ?? input;
        var giC = giOwned ?? gradInput;

        int n = checked((int)goC.ElementCount);
        float* pGo = goC.DataPointer;
        float* pIn = inC.DataPointer;
        float* pGi = giC.DataPointer;

        if (n >= ParallelElementThreshold)
        {
            ParallelChunked(n, (start, len) => ReluBackwardChunk(pGo + start, pIn + start, pGi + start, len));
        }
        else
        {
            ReluBackwardChunk(pGo, pIn, pGi, n);
        }

        if (giOwned != null)
        {
            Tensor<float>.CopyElements(giOwned, gradInput);
        }
    }

    public static void Sigmoid(Tensor<float> input, Tensor<float> output)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        if (input.ElementCount != output.ElementCount)
            throw new ArgumentException("Tensor element counts must match.");

        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var outOwned = (output.IsContiguous && output.ElementOffset == 0) ? null : new Tensor<float>(output.Shape);
        var inC = inOwned ?? input;
        var outC = outOwned ?? output;

        int n = checked((int)inC.ElementCount);
        float* pIn = inC.DataPointer;
        float* pOut = outC.DataPointer;

        if (n >= ParallelElementThreshold)
        {
            ParallelChunked(n, (start, len) => SigmoidChunk(pIn + start, pOut + start, len));
        }
        else
        {
            SigmoidChunk(pIn, pOut, n);
        }

        if (outOwned != null)
        {
            Tensor<float>.CopyElements(outOwned, output);
        }
    }

    public static void Softmax(Tensor<float> input, Tensor<float> output, int dim = -1)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        if (dim < 0) dim += input.Rank;
        if ((uint)dim >= (uint)input.Rank)
            throw new ArgumentOutOfRangeException(nameof(dim));
        if (!input.Shape.Equals(output.Shape))
            throw new ArgumentException("Input and output shapes must match for Softmax.");

        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var outOwned = (output.IsContiguous && output.ElementOffset == 0) ? null : new Tensor<float>(output.Shape);
        var inC = inOwned ?? input;
        var outC = outOwned ?? output;

        long rowsL = 1;
        for (int i = 0; i < dim; i++) rowsL = checked(rowsL * inC.Shape[i]);
        int cols = inC.Shape[dim];
        long innerL = 1;
        for (int i = dim + 1; i < inC.Rank; i++) innerL = checked(innerL * inC.Shape[i]);

        int rows = checked((int)rowsL);
        int inner = checked((int)innerL);
        if (rows == 0 || cols == 0 || inner == 0)
        {
            return;
        }

        float* pIn = inC.DataPointer;
        float* pOut = outC.DataPointer;

        if (inner == 1)
        {
            if ((long)rows * cols >= 16384 && rows > 1)
            {
                Parallel.For(0, rows, r => SoftmaxRowSimd(pIn + (long)r * cols, pOut + (long)r * cols, cols));
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    SoftmaxRowSimd(pIn + (long)r * cols, pOut + (long)r * cols, cols);
                }
            }
        }
        else
        {
            long outerStride = (long)cols * inner;
            if ((long)rows * cols * inner >= 16384 && rows > 1)
            {
                Parallel.For(0, rows, r =>
                {
                    long baseOffset = r * outerStride;
                    for (int k = 0; k < inner; k++)
                    {
                        SoftmaxStridedLane(pIn, pOut, baseOffset + k, cols, inner);
                    }
                });
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    long baseOffset = r * outerStride;
                    for (int k = 0; k < inner; k++)
                    {
                        SoftmaxStridedLane(pIn, pOut, baseOffset + k, cols, inner);
                    }
                }
            }
        }

        if (outOwned != null)
        {
            Tensor<float>.CopyElements(outOwned, output);
        }
    }

    public static void SoftmaxBackward(
        Tensor<float> gradOutput,
        Tensor<float> softmaxOutput,
        Tensor<float> gradInput,
        int dim = -1)
    {
        ArgumentNullException.ThrowIfNull(gradOutput);
        ArgumentNullException.ThrowIfNull(softmaxOutput);
        ArgumentNullException.ThrowIfNull(gradInput);
        if (dim < 0) dim += softmaxOutput.Rank;
        if ((uint)dim >= (uint)softmaxOutput.Rank)
            throw new ArgumentOutOfRangeException(nameof(dim));

        using var goOwned = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0) ? null : gradOutput.Contiguous();
        using var yOwned = (softmaxOutput.IsContiguous && softmaxOutput.ElementOffset == 0) ? null : softmaxOutput.Contiguous();
        using var giOwned = (gradInput.IsContiguous && gradInput.ElementOffset == 0) ? null : new Tensor<float>(gradInput.Shape);
        var goC = goOwned ?? gradOutput;
        var yC = yOwned ?? softmaxOutput;
        var giC = giOwned ?? gradInput;

        long rowsL = 1;
        for (int i = 0; i < dim; i++) rowsL = checked(rowsL * yC.Shape[i]);
        int cols = yC.Shape[dim];
        long innerL = 1;
        for (int i = dim + 1; i < yC.Rank; i++) innerL = checked(innerL * yC.Shape[i]);

        int rows = checked((int)rowsL);
        int inner = checked((int)innerL);
        if (rows == 0 || cols == 0 || inner == 0)
        {
            return;
        }

        float* pGo = goC.DataPointer;
        float* pY = yC.DataPointer;
        float* pGi = giC.DataPointer;

        if (inner == 1)
        {
            if ((long)rows * cols >= 16384 && rows > 1)
            {
                Parallel.For(0, rows, r => SoftmaxBackwardRowSimd(pGo + (long)r * cols, pY + (long)r * cols, pGi + (long)r * cols, cols));
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    SoftmaxBackwardRowSimd(pGo + (long)r * cols, pY + (long)r * cols, pGi + (long)r * cols, cols);
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
                    double dot = 0.0;
                    for (int c = 0; c < cols; c++)
                    {
                        long idx = offset + (long)c * inner;
                        dot += (double)pGo[idx] * pY[idx];
                    }

                    float dotF = (float)dot;
                    for (int c = 0; c < cols; c++)
                    {
                        long idx = offset + (long)c * inner;
                        pGi[idx] = pY[idx] * (pGo[idx] - dotF);
                    }
                }
            }
        }

        if (giOwned != null)
        {
            Tensor<float>.CopyElements(giOwned, gradInput);
        }
    }

    public static void LayerNorm(
        Tensor<float> input,
        Tensor<float>? gamma,
        Tensor<float>? beta,
        Tensor<float> output,
        float eps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        if (input.Rank == 0)
            throw new ArgumentException("LayerNorm requires Rank >= 1.");

        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var outOwned = (output.IsContiguous && output.ElementOffset == 0) ? null : new Tensor<float>(output.Shape);
        using var gOwned = (gamma == null || (gamma.IsContiguous && gamma.ElementOffset == 0)) ? null : gamma.Contiguous();
        using var bOwned = (beta == null || (beta.IsContiguous && beta.ElementOffset == 0)) ? null : beta.Contiguous();

        var inC = inOwned ?? input;
        var outC = outOwned ?? output;
        var gC = gOwned ?? gamma;
        var bC = bOwned ?? beta;

        int d = inC.Shape[inC.Rank - 1];
        if (d == 0 || inC.ElementCount == 0)
        {
            return;
        }

        int batch = checked((int)(inC.ElementCount / d));
        float* pIn = inC.DataPointer;
        float* pOut = outC.DataPointer;
        float* pG = gC != null ? gC.DataPointer : null;
        float* pB = bC != null ? bC.DataPointer : null;

        if ((long)batch * d >= 16384 && batch > 1)
        {
            Parallel.For(0, batch, b =>
            {
                long offset = (long)b * d;
                LayerNormRowSimd(pIn + offset, pG, pB, pOut + offset, d, eps);
            });
        }
        else
        {
            for (int b = 0; b < batch; b++)
            {
                long offset = (long)b * d;
                LayerNormRowSimd(pIn + offset, pG, pB, pOut + offset, d, eps);
            }
        }

        if (outOwned != null)
        {
            Tensor<float>.CopyElements(outOwned, output);
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
        ArgumentNullException.ThrowIfNull(gradOutput);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gradInput);
        if (input.Rank == 0)
            throw new ArgumentException("LayerNormBackward requires Rank >= 1.");

        using var goOwned = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0) ? null : gradOutput.Contiguous();
        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var gOwned = (gamma == null || (gamma.IsContiguous && gamma.ElementOffset == 0)) ? null : gamma.Contiguous();
        using var giOwned = (gradInput.IsContiguous && gradInput.ElementOffset == 0) ? null : new Tensor<float>(gradInput.Shape);
        using var ggOwned = (gradGamma == null || (gradGamma.IsContiguous && gradGamma.ElementOffset == 0)) ? null : gradGamma.Contiguous();
        using var gbOwned = (gradBeta == null || (gradBeta.IsContiguous && gradBeta.ElementOffset == 0)) ? null : gradBeta.Contiguous();

        var goC = goOwned ?? gradOutput;
        var inC = inOwned ?? input;
        var gC = gOwned ?? gamma;
        var giC = giOwned ?? gradInput;
        var ggC = ggOwned ?? gradGamma;
        var gbC = gbOwned ?? gradBeta;

        int d = inC.Shape[inC.Rank - 1];
        if (d == 0 || inC.ElementCount == 0)
        {
            return;
        }

        int batch = checked((int)(inC.ElementCount / d));
        float* pGo = goC.DataPointer;
        float* pIn = inC.DataPointer;
        float* pG = gC != null ? gC.DataPointer : null;
        float* pGi = giC.DataPointer;
        float* pGg = ggC != null ? ggC.DataPointer : null;
        float* pGb = gbC != null ? gbC.DataPointer : null;

        if (pGg != null) new Span<float>(pGg, d).Clear();
        if (pGb != null) new Span<float>(pGb, d).Clear();

        if ((long)batch * d >= 16384 && batch > 1 && pGg == null && pGb == null)
        {
            Parallel.For(0, batch, b =>
            {
                long offset = (long)b * d;
                LayerNormBackwardRowSimd(pGo + offset, pIn + offset, pG, pGi + offset, null, null, d, eps);
            });
        }
        else
        {
            for (int b = 0; b < batch; b++)
            {
                long offset = (long)b * d;
                LayerNormBackwardRowSimd(pGo + offset, pIn + offset, pG, pGi + offset, pGg, pGb, d, eps);
            }
        }

        if (giOwned != null)
        {
            Tensor<float>.CopyElements(giOwned, gradInput);
        }
        if (gradGamma != null && ggOwned != null)
        {
            Tensor<float>.CopyElements(ggOwned, gradGamma);
        }
        if (gradBeta != null && gbOwned != null)
        {
            Tensor<float>.CopyElements(gbOwned, gradBeta);
        }
    }

    public static void GELU(Tensor<float> input, Tensor<float> output)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        if (input.ElementCount != output.ElementCount)
            throw new ArgumentException("Tensor element counts must match.");

        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var outOwned = (output.IsContiguous && output.ElementOffset == 0) ? null : new Tensor<float>(output.Shape);
        var inC = inOwned ?? input;
        var outC = outOwned ?? output;

        int n = checked((int)inC.ElementCount);
        float* pIn = inC.DataPointer;
        float* pOut = outC.DataPointer;

        if (n >= 16384)
        {
            ParallelChunked(n, (start, len) => GeluChunk(pIn + start, pOut + start, len));
        }
        else
        {
            GeluChunk(pIn, pOut, n);
        }

        if (outOwned != null)
        {
            Tensor<float>.CopyElements(outOwned, output);
        }
    }

    public static void GELUBackward(Tensor<float> gradOutput, Tensor<float> input, Tensor<float> gradInput)
    {
        ArgumentNullException.ThrowIfNull(gradOutput);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gradInput);
        if (gradOutput.ElementCount != input.ElementCount || input.ElementCount != gradInput.ElementCount)
            throw new ArgumentException("Tensor element counts must match.");

        using var goOwned = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0) ? null : gradOutput.Contiguous();
        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var giOwned = (gradInput.IsContiguous && gradInput.ElementOffset == 0) ? null : new Tensor<float>(gradInput.Shape);
        var goC = goOwned ?? gradOutput;
        var inC = inOwned ?? input;
        var giC = giOwned ?? gradInput;

        int n = checked((int)goC.ElementCount);
        float* pGo = goC.DataPointer;
        float* pIn = inC.DataPointer;
        float* pGi = giC.DataPointer;

        if (n >= 16384)
        {
            ParallelChunked(n, (start, len) => GeluBackwardChunk(pGo + start, pIn + start, pGi + start, len));
        }
        else
        {
            GeluBackwardChunk(pGo, pIn, pGi, n);
        }

        if (giOwned != null)
        {
            Tensor<float>.CopyElements(giOwned, gradInput);
        }
    }

    public static void RMSNorm(Tensor<float> input, Tensor<float>? weight, Tensor<float> output, float eps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        if (input.Rank == 0)
            throw new ArgumentException("RMSNorm requires Rank >= 1.");

        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var outOwned = (output.IsContiguous && output.ElementOffset == 0) ? null : new Tensor<float>(output.Shape);
        using var wOwned = (weight == null || (weight.IsContiguous && weight.ElementOffset == 0)) ? null : weight.Contiguous();

        var inC = inOwned ?? input;
        var outC = outOwned ?? output;
        var wC = wOwned ?? weight;

        int d = inC.Shape[inC.Rank - 1];
        if (d == 0 || inC.ElementCount == 0)
        {
            return;
        }

        int batch = checked((int)(inC.ElementCount / d));
        float* pIn = inC.DataPointer;
        float* pOut = outC.DataPointer;
        float* pW = wC != null ? wC.DataPointer : null;

        if ((long)batch * d >= 16384 && batch > 1)
        {
            Parallel.For(0, batch, b =>
            {
                long offset = (long)b * d;
                RmsNormRowSimd(pIn + offset, pW, pOut + offset, d, eps);
            });
        }
        else
        {
            for (int b = 0; b < batch; b++)
            {
                long offset = (long)b * d;
                RmsNormRowSimd(pIn + offset, pW, pOut + offset, d, eps);
            }
        }

        if (outOwned != null)
        {
            Tensor<float>.CopyElements(outOwned, output);
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
        ArgumentNullException.ThrowIfNull(gradOutput);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gradInput);
        if (input.Rank == 0)
            throw new ArgumentException("RMSNormBackward requires Rank >= 1.");

        using var goOwned = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0) ? null : gradOutput.Contiguous();
        using var inOwned = (input.IsContiguous && input.ElementOffset == 0) ? null : input.Contiguous();
        using var wOwned = (weight == null || (weight.IsContiguous && weight.ElementOffset == 0)) ? null : weight.Contiguous();
        using var giOwned = (gradInput.IsContiguous && gradInput.ElementOffset == 0) ? null : new Tensor<float>(gradInput.Shape);
        using var gwOwned = (gradWeight == null || (gradWeight.IsContiguous && gradWeight.ElementOffset == 0)) ? null : gradWeight.Contiguous();

        var goC = goOwned ?? gradOutput;
        var inC = inOwned ?? input;
        var wC = wOwned ?? weight;
        var giC = giOwned ?? gradInput;
        var gwC = gwOwned ?? gradWeight;

        int d = inC.Shape[inC.Rank - 1];
        if (d == 0 || inC.ElementCount == 0)
        {
            return;
        }

        int batch = checked((int)(inC.ElementCount / d));
        float* pGo = goC.DataPointer;
        float* pIn = inC.DataPointer;
        float* pW = wC != null ? wC.DataPointer : null;
        float* pGi = giC.DataPointer;
        float* pGw = gwC != null ? gwC.DataPointer : null;

        if (pGw != null)
        {
            new Span<float>(pGw, d).Clear();
        }

        if ((long)batch * d >= 16384 && batch > 1 && pGw == null)
        {
            Parallel.For(0, batch, b =>
            {
                long offset = (long)b * d;
                RmsNormBackwardRowSimd(pGo + offset, pIn + offset, pW, pGi + offset, null, d, eps);
            });
        }
        else
        {
            for (int b = 0; b < batch; b++)
            {
                long offset = (long)b * d;
                RmsNormBackwardRowSimd(pGo + offset, pIn + offset, pW, pGi + offset, pGw, d, eps);
            }
        }

        if (giOwned != null)
        {
            Tensor<float>.CopyElements(giOwned, gradInput);
        }
        if (gradWeight != null && gwOwned != null)
        {
            Tensor<float>.CopyElements(gwOwned, gradWeight);
        }
    }

    private static void ExecuteBinary<TOp>(Tensor<float> a, Tensor<float> b, Tensor<float> c)
        where TOp : struct, IBinaryOp
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        if (a.ElementCount != b.ElementCount || a.ElementCount != c.ElementCount)
            throw new ArgumentException("Tensor element counts must match for elementwise operations.");

        using var aOwned = (a.IsContiguous && a.ElementOffset == 0) ? null : a.Contiguous();
        using var bOwned = (b.IsContiguous && b.ElementOffset == 0) ? null : b.Contiguous();
        using var cOwned = (c.IsContiguous && c.ElementOffset == 0) ? null : new Tensor<float>(c.Shape);

        var aSrc = aOwned ?? a;
        var bSrc = bOwned ?? b;
        var cDst = cOwned ?? c;

        int n = checked((int)aSrc.ElementCount);
        float* pA = aSrc.DataPointer;
        float* pB = bSrc.DataPointer;
        float* pC = cDst.DataPointer;

        if (n >= ParallelElementThreshold)
        {
            ParallelChunked(n, (start, len) => BinaryChunk<TOp>(pA + start, pB + start, pC + start, len));
        }
        else
        {
            BinaryChunk<TOp>(pA, pB, pC, n);
        }

        if (cOwned != null)
        {
            Tensor<float>.CopyElements(cOwned, c);
        }
    }

    private static void ParallelChunked(int n, Action<int, int> body)
    {
        int workers = Math.Clamp(Environment.ProcessorCount, 2, 16);
        int chunk = (n + workers - 1) / workers;
        Parallel.For(0, workers, w =>
        {
            int start = w * chunk;
            if (start >= n) return;
            int len = Math.Min(chunk, n - start);
            body(start, len);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BinaryChunk<TOp>(float* a, float* b, float* c, int n)
        where TOp : struct, IBinaryOp
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                var va = Vector512.Load(a + i);
                var vb = Vector512.Load(b + i);
                Vector512.Store(TOp.Apply(va, vb), c + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                Vector256.Store(TOp.Apply(va, vb), c + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                var va = Vector128.Load(a + i);
                var vb = Vector128.Load(b + i);
                Vector128.Store(TOp.Apply(va, vb), c + i);
            }
        }

        for (; i < n; i++)
        {
            c[i] = TOp.Apply(a[i], b[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ScaleChunk(float* a, float* c, float scalar, int n)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vs = Vector512.Create(scalar);
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                Vector512.Store(Vector512.Load(a + i) * vs, c + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vs = Vector256.Create(scalar);
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                Vector256.Store(Vector256.Load(a + i) * vs, c + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vs = Vector128.Create(scalar);
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                Vector128.Store(Vector128.Load(a + i) * vs, c + i);
            }
        }

        for (; i < n; i++)
        {
            c[i] = a[i] * scalar;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ReluChunk(float* src, float* dst, int n)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vZero = Vector512<float>.Zero;
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                Vector512.Store(Vector512.Max(Vector512.Load(src + i), vZero), dst + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vZero = Vector256<float>.Zero;
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                Vector256.Store(Vector256.Max(Vector256.Load(src + i), vZero), dst + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vZero = Vector128<float>.Zero;
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                Vector128.Store(Vector128.Max(Vector128.Load(src + i), vZero), dst + i);
            }
        }

        for (; i < n; i++)
        {
            float v = src[i];
            dst[i] = v > 0.0f ? v : 0.0f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ReluBackwardChunk(float* go, float* inp, float* gi, int n)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vZero = Vector512<float>.Zero;
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                var mask = Vector512.GreaterThan(Vector512.Load(inp + i), vZero);
                Vector512.Store(Vector512.ConditionalSelect(mask, Vector512.Load(go + i), vZero), gi + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vZero = Vector256<float>.Zero;
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                var mask = Vector256.GreaterThan(Vector256.Load(inp + i), vZero);
                Vector256.Store(Vector256.ConditionalSelect(mask, Vector256.Load(go + i), vZero), gi + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vZero = Vector128<float>.Zero;
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                var mask = Vector128.GreaterThan(Vector128.Load(inp + i), vZero);
                Vector128.Store(Vector128.ConditionalSelect(mask, Vector128.Load(go + i), vZero), gi + i);
            }
        }

        for (; i < n; i++)
        {
            gi[i] = inp[i] > 0.0f ? go[i] : 0.0f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SigmoidChunk(float* src, float* dst, int n)
    {
        // Numerically stable across (-inf, +inf):
        // e = exp(-|x|); result = (x >= 0 ? 1 : e) / (1 + e)
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vZero = Vector512<float>.Zero;
            var vOne = Vector512.Create(1.0f);
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                var vx = Vector512.Load(src + i);
                var negAbsX = Vector512.Negate(Vector512.Abs(vx));
                var e = SimdMath.FastExpNonPositiveAvx512(negAbsX);
                var num = Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(vx, vZero), vOne, e);
                Vector512.Store(num / (vOne + e), dst + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vZero = Vector256<float>.Zero;
            var vOne = Vector256.Create(1.0f);
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                var vx = Vector256.Load(src + i);
                var negAbsX = Vector256.Negate(Vector256.Abs(vx));
                var e = SimdMath.FastExpNonPositiveAvx2(negAbsX);
                var num = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(vx, vZero), vOne, e);
                Vector256.Store(num / (vOne + e), dst + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vZero = Vector128<float>.Zero;
            var vOne = Vector128.Create(1.0f);
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                var vx = Vector128.Load(src + i);
                var negAbsX = Vector128.Negate(Vector128.Abs(vx));
                var e = SimdMath.FastExpNonPositiveAdvSimd(negAbsX);
                var num = Vector128.ConditionalSelect(Vector128.GreaterThanOrEqual(vx, vZero), vOne, e);
                Vector128.Store(num / (vOne + e), dst + i);
            }
        }

        for (; i < n; i++)
        {
            float x = src[i];
            float e = SimdMath.FastExpNonPositiveScalar(-MathF.Abs(x));
            dst[i] = (x >= 0.0f ? 1.0f : e) / (1.0f + e);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GeluChunk(float* src, float* dst, int n)
    {
        const float sqrt2OverPi = 0.7978845608f;
        const float coeff = 0.044715f;

        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vS2P = Vector512.Create(sqrt2OverPi);
            var vCoeff = Vector512.Create(coeff);
            var vHalf = Vector512.Create(0.5f);
            var vOne = Vector512.Create(1.0f);
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                var x = Vector512.Load(src + i);
                var x3 = x * x * x;
                var u = vS2P * (x + vCoeff * x3);
                var t = SimdMath.FastTanhAvx512(u);
                Vector512.Store(vHalf * x * (vOne + t), dst + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vS2P = Vector256.Create(sqrt2OverPi);
            var vCoeff = Vector256.Create(coeff);
            var vHalf = Vector256.Create(0.5f);
            var vOne = Vector256.Create(1.0f);
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                var x = Vector256.Load(src + i);
                var x3 = x * x * x;
                var u = vS2P * (x + vCoeff * x3);
                var t = SimdMath.FastTanhAvx2(u);
                Vector256.Store(vHalf * x * (vOne + t), dst + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vS2P = Vector128.Create(sqrt2OverPi);
            var vCoeff = Vector128.Create(coeff);
            var vHalf = Vector128.Create(0.5f);
            var vOne = Vector128.Create(1.0f);
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                var x = Vector128.Load(src + i);
                var x3 = x * x * x;
                var u = vS2P * (x + vCoeff * x3);
                var t = SimdMath.FastTanhAdvSimd(u);
                Vector128.Store(vHalf * x * (vOne + t), dst + i);
            }
        }

        for (; i < n; i++)
        {
            float x = src[i];
            float u = sqrt2OverPi * (x + coeff * x * x * x);
            float t = SimdMath.FastTanhScalar(u);
            dst[i] = 0.5f * x * (1.0f + t);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GeluBackwardChunk(float* go, float* inp, float* gi, int n)
    {
        const float sqrt2OverPi = 0.7978845608f;
        const float coeff = 0.044715f;
        const float coeff3 = 3.0f * coeff;

        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vS2P = Vector512.Create(sqrt2OverPi);
            var vCoeff = Vector512.Create(coeff);
            var vCoeff3 = Vector512.Create(coeff3);
            var vHalf = Vector512.Create(0.5f);
            var vOne = Vector512.Create(1.0f);
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                var x = Vector512.Load(inp + i);
                var x2 = x * x;
                var u = vS2P * (x + vCoeff * x * x2);
                var t = SimdMath.FastTanhAvx512(u);
                var oneMinusT2 = vOne - t * t;
                var duDx = vS2P * (vOne + vCoeff3 * x2);
                var dGelu = vHalf * (vOne + t) + vHalf * x * oneMinusT2 * duDx;
                Vector512.Store(Vector512.Load(go + i) * dGelu, gi + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vS2P = Vector256.Create(sqrt2OverPi);
            var vCoeff = Vector256.Create(coeff);
            var vCoeff3 = Vector256.Create(coeff3);
            var vHalf = Vector256.Create(0.5f);
            var vOne = Vector256.Create(1.0f);
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                var x = Vector256.Load(inp + i);
                var x2 = x * x;
                var u = vS2P * (x + vCoeff * x * x2);
                var t = SimdMath.FastTanhAvx2(u);
                var oneMinusT2 = vOne - t * t;
                var duDx = vS2P * (vOne + vCoeff3 * x2);
                var dGelu = vHalf * (vOne + t) + vHalf * x * oneMinusT2 * duDx;
                Vector256.Store(Vector256.Load(go + i) * dGelu, gi + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vS2P = Vector128.Create(sqrt2OverPi);
            var vCoeff = Vector128.Create(coeff);
            var vCoeff3 = Vector128.Create(coeff3);
            var vHalf = Vector128.Create(0.5f);
            var vOne = Vector128.Create(1.0f);
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                var x = Vector128.Load(inp + i);
                var x2 = x * x;
                var u = vS2P * (x + vCoeff * x * x2);
                var t = SimdMath.FastTanhAdvSimd(u);
                var oneMinusT2 = vOne - t * t;
                var duDx = vS2P * (vOne + vCoeff3 * x2);
                var dGelu = vHalf * (vOne + t) + vHalf * x * oneMinusT2 * duDx;
                Vector128.Store(Vector128.Load(go + i) * dGelu, gi + i);
            }
        }

        for (; i < n; i++)
        {
            float x = inp[i];
            float x2 = x * x;
            float u = sqrt2OverPi * (x + coeff * x * x2);
            float t = SimdMath.FastTanhScalar(u);
            float oneMinusT2 = 1.0f - t * t;
            float duDx = sqrt2OverPi * (1.0f + coeff3 * x2);
            float dGelu = 0.5f * (1.0f + t) + 0.5f * x * oneMinusT2 * duDx;
            gi[i] = go[i] * dGelu;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SoftmaxRowSimd(float* rowIn, float* rowOut, int cols)
    {
        float maxVal = FindMaxSimd(rowIn, cols);

        double sumExp = 0.0;
        int c = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var vMax = Vector512.Create(maxVal);
            var accLo = Vector512<double>.Zero;
            var accHi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            for (; c + step <= cols; c += step)
            {
                var e = SimdMath.FastExpNonPositiveAvx512(Vector512.Load(rowIn + c) - vMax);
                Vector512.Store(e, rowOut + c);
                var (lo, hi) = Vector512.Widen(e);
                accLo += lo;
                accHi += hi;
            }
            sumExp += Vector512.Sum(accLo + accHi);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vMax = Vector256.Create(maxVal);
            var accLo = Vector256<double>.Zero;
            var accHi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            for (; c + step <= cols; c += step)
            {
                var e = SimdMath.FastExpNonPositiveAvx2(Vector256.Load(rowIn + c) - vMax);
                Vector256.Store(e, rowOut + c);
                var (lo, hi) = Vector256.Widen(e);
                accLo += lo;
                accHi += hi;
            }
            sumExp += Vector256.Sum(accLo + accHi);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vMax = Vector128.Create(maxVal);
            var accLo = Vector128<double>.Zero;
            var accHi = Vector128<double>.Zero;
            int step = Vector128<float>.Count;
            for (; c + step <= cols; c += step)
            {
                var e = SimdMath.FastExpNonPositiveAdvSimd(Vector128.Load(rowIn + c) - vMax);
                Vector128.Store(e, rowOut + c);
                var (lo, hi) = Vector128.Widen(e);
                accLo += lo;
                accHi += hi;
            }
            sumExp += Vector128.Sum(accLo + accHi);
        }

        for (; c < cols; c++)
        {
            float e = SimdMath.FastExpNonPositiveScalar(rowIn[c] - maxVal);
            rowOut[c] = e;
            sumExp += e;
        }

        float invSum = (float)(1.0 / sumExp);
        ScaleChunk(rowOut, rowOut, invSum, cols);
    }

    private static void SoftmaxStridedLane(float* pIn, float* pOut, long offset, int cols, int inner)
    {
        float maxVal = pIn[offset];
        for (int c = 1; c < cols; c++)
        {
            float val = pIn[offset + (long)c * inner];
            if (val > maxVal) maxVal = val;
        }

        double sumExp = 0.0;
        for (int c = 0; c < cols; c++)
        {
            long idx = offset + (long)c * inner;
            float e = SimdMath.FastExpNonPositiveScalar(pIn[idx] - maxVal);
            pOut[idx] = e;
            sumExp += e;
        }

        float invSum = (float)(1.0 / sumExp);
        for (int c = 0; c < cols; c++)
        {
            pOut[offset + (long)c * inner] *= invSum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SoftmaxBackwardRowSimd(float* rowGo, float* rowY, float* rowGi, int cols)
    {
        double dot = 0.0;
        int c = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var accLo = Vector512<double>.Zero;
            var accHi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            for (; c + step <= cols; c += step)
            {
                var prod = Vector512.Load(rowGo + c) * Vector512.Load(rowY + c);
                var (lo, hi) = Vector512.Widen(prod);
                accLo += lo;
                accHi += hi;
            }
            dot += Vector512.Sum(accLo + accHi);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var accLo = Vector256<double>.Zero;
            var accHi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            for (; c + step <= cols; c += step)
            {
                var prod = Vector256.Load(rowGo + c) * Vector256.Load(rowY + c);
                var (lo, hi) = Vector256.Widen(prod);
                accLo += lo;
                accHi += hi;
            }
            dot += Vector256.Sum(accLo + accHi);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var accLo = Vector128<double>.Zero;
            var accHi = Vector128<double>.Zero;
            int step = Vector128<float>.Count;
            for (; c + step <= cols; c += step)
            {
                var prod = Vector128.Load(rowGo + c) * Vector128.Load(rowY + c);
                var (lo, hi) = Vector128.Widen(prod);
                accLo += lo;
                accHi += hi;
            }
            dot += Vector128.Sum(accLo + accHi);
        }

        for (; c < cols; c++)
        {
            dot += (double)rowGo[c] * rowY[c];
        }

        float dotF = (float)dot;
        c = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var vDot = Vector512.Create(dotF);
            int step = Vector512<float>.Count;
            for (; c + step <= cols; c += step)
            {
                Vector512.Store(Vector512.Load(rowY + c) * (Vector512.Load(rowGo + c) - vDot), rowGi + c);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vDot = Vector256.Create(dotF);
            int step = Vector256<float>.Count;
            for (; c + step <= cols; c += step)
            {
                Vector256.Store(Vector256.Load(rowY + c) * (Vector256.Load(rowGo + c) - vDot), rowGi + c);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vDot = Vector128.Create(dotF);
            int step = Vector128<float>.Count;
            for (; c + step <= cols; c += step)
            {
                Vector128.Store(Vector128.Load(rowY + c) * (Vector128.Load(rowGo + c) - vDot), rowGi + c);
            }
        }

        for (; c < cols; c++)
        {
            rowGi[c] = rowY[c] * (rowGo[c] - dotF);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void LayerNormRowSimd(float* rowIn, float* pG, float* pB, float* rowOut, int d, float eps)
    {
        float mean = (float)(SimdMath.WidenSum(rowIn, d) / d);
        double varSum = SumSquaredDiffSimd(rowIn, mean, d);
        float invStd = 1.0f / MathF.Sqrt((float)(varSum / d) + eps);

        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vMean = Vector512.Create(mean);
            var vInvStd = Vector512.Create(invStd);
            int step = Vector512<float>.Count;
            for (; i + step <= d; i += step)
            {
                var norm = (Vector512.Load(rowIn + i) - vMean) * vInvStd;
                if (pG != null) norm *= Vector512.Load(pG + i);
                if (pB != null) norm += Vector512.Load(pB + i);
                Vector512.Store(norm, rowOut + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vMean = Vector256.Create(mean);
            var vInvStd = Vector256.Create(invStd);
            int step = Vector256<float>.Count;
            for (; i + step <= d; i += step)
            {
                var norm = (Vector256.Load(rowIn + i) - vMean) * vInvStd;
                if (pG != null) norm *= Vector256.Load(pG + i);
                if (pB != null) norm += Vector256.Load(pB + i);
                Vector256.Store(norm, rowOut + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vMean = Vector128.Create(mean);
            var vInvStd = Vector128.Create(invStd);
            int step = Vector128<float>.Count;
            for (; i + step <= d; i += step)
            {
                var norm = (Vector128.Load(rowIn + i) - vMean) * vInvStd;
                if (pG != null) norm *= Vector128.Load(pG + i);
                if (pB != null) norm += Vector128.Load(pB + i);
                Vector128.Store(norm, rowOut + i);
            }
        }

        for (; i < d; i++)
        {
            float norm = (rowIn[i] - mean) * invStd;
            if (pG != null) norm *= pG[i];
            if (pB != null) norm += pB[i];
            rowOut[i] = norm;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void LayerNormBackwardRowSimd(
        float* rowGo,
        float* rowIn,
        float* pG,
        float* rowGi,
        float* pGg,
        float* pGb,
        int d,
        float eps)
    {
        float invD = 1.0f / d;
        float mean = (float)(SimdMath.WidenSum(rowIn, d) * invD);
        double varSum = SumSquaredDiffSimd(rowIn, mean, d);
        float invStd = 1.0f / MathF.Sqrt((float)(varSum * invD) + eps);

        double sumDxHat = 0.0;
        double sumDxHatXHat = 0.0;
        int i = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var vMean = Vector512.Create(mean);
            var vInvStd = Vector512.Create(invStd);
            var acc1Lo = Vector512<double>.Zero;
            var acc1Hi = Vector512<double>.Zero;
            var acc2Lo = Vector512<double>.Zero;
            var acc2Hi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector512.Load(rowGo + i);
                var xHat = (Vector512.Load(rowIn + i) - vMean) * vInvStd;
                var g = pG != null ? dy * Vector512.Load(pG + i) : dy;
                var (gLo, gHi) = Vector512.Widen(g);
                acc1Lo += gLo;
                acc1Hi += gHi;
                var (gxLo, gxHi) = Vector512.Widen(g * xHat);
                acc2Lo += gxLo;
                acc2Hi += gxHi;
                if (pGg != null) Vector512.Store(Vector512.Load(pGg + i) + dy * xHat, pGg + i);
                if (pGb != null) Vector512.Store(Vector512.Load(pGb + i) + dy, pGb + i);
            }
            sumDxHat += Vector512.Sum(acc1Lo + acc1Hi);
            sumDxHatXHat += Vector512.Sum(acc2Lo + acc2Hi);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vMean = Vector256.Create(mean);
            var vInvStd = Vector256.Create(invStd);
            var acc1Lo = Vector256<double>.Zero;
            var acc1Hi = Vector256<double>.Zero;
            var acc2Lo = Vector256<double>.Zero;
            var acc2Hi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector256.Load(rowGo + i);
                var xHat = (Vector256.Load(rowIn + i) - vMean) * vInvStd;
                var g = pG != null ? dy * Vector256.Load(pG + i) : dy;
                var (gLo, gHi) = Vector256.Widen(g);
                acc1Lo += gLo;
                acc1Hi += gHi;
                var (gxLo, gxHi) = Vector256.Widen(g * xHat);
                acc2Lo += gxLo;
                acc2Hi += gxHi;
                if (pGg != null) Vector256.Store(Vector256.Load(pGg + i) + dy * xHat, pGg + i);
                if (pGb != null) Vector256.Store(Vector256.Load(pGb + i) + dy, pGb + i);
            }
            sumDxHat += Vector256.Sum(acc1Lo + acc1Hi);
            sumDxHatXHat += Vector256.Sum(acc2Lo + acc2Hi);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vMean = Vector128.Create(mean);
            var vInvStd = Vector128.Create(invStd);
            var acc1Lo = Vector128<double>.Zero;
            var acc1Hi = Vector128<double>.Zero;
            var acc2Lo = Vector128<double>.Zero;
            var acc2Hi = Vector128<double>.Zero;
            int step = Vector128<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector128.Load(rowGo + i);
                var xHat = (Vector128.Load(rowIn + i) - vMean) * vInvStd;
                var g = pG != null ? dy * Vector128.Load(pG + i) : dy;
                var (gLo, gHi) = Vector128.Widen(g);
                acc1Lo += gLo;
                acc1Hi += gHi;
                var (gxLo, gxHi) = Vector128.Widen(g * xHat);
                acc2Lo += gxLo;
                acc2Hi += gxHi;
                if (pGg != null) Vector128.Store(Vector128.Load(pGg + i) + dy * xHat, pGg + i);
                if (pGb != null) Vector128.Store(Vector128.Load(pGb + i) + dy, pGb + i);
            }
            sumDxHat += Vector128.Sum(acc1Lo + acc1Hi);
            sumDxHatXHat += Vector128.Sum(acc2Lo + acc2Hi);
        }

        for (; i < d; i++)
        {
            float dy = rowGo[i];
            float xHat = (rowIn[i] - mean) * invStd;
            float g = pG != null ? dy * pG[i] : dy;
            sumDxHat += g;
            sumDxHatXHat += g * xHat;
            if (pGg != null) pGg[i] += dy * xHat;
            if (pGb != null) pGb[i] += dy;
        }

        float c1 = invD * (float)sumDxHat;
        float c2 = invD * (float)sumDxHatXHat;
        i = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var vMean = Vector512.Create(mean);
            var vInvStd = Vector512.Create(invStd);
            var vC1 = Vector512.Create(c1);
            var vC2 = Vector512.Create(c2);
            int step = Vector512<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector512.Load(rowGo + i);
                var xHat = (Vector512.Load(rowIn + i) - vMean) * vInvStd;
                var g = pG != null ? dy * Vector512.Load(pG + i) : dy;
                Vector512.Store(vInvStd * (g - vC1 - xHat * vC2), rowGi + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vMean = Vector256.Create(mean);
            var vInvStd = Vector256.Create(invStd);
            var vC1 = Vector256.Create(c1);
            var vC2 = Vector256.Create(c2);
            int step = Vector256<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector256.Load(rowGo + i);
                var xHat = (Vector256.Load(rowIn + i) - vMean) * vInvStd;
                var g = pG != null ? dy * Vector256.Load(pG + i) : dy;
                Vector256.Store(vInvStd * (g - vC1 - xHat * vC2), rowGi + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vMean = Vector128.Create(mean);
            var vInvStd = Vector128.Create(invStd);
            var vC1 = Vector128.Create(c1);
            var vC2 = Vector128.Create(c2);
            int step = Vector128<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector128.Load(rowGo + i);
                var xHat = (Vector128.Load(rowIn + i) - vMean) * vInvStd;
                var g = pG != null ? dy * Vector128.Load(pG + i) : dy;
                Vector128.Store(vInvStd * (g - vC1 - xHat * vC2), rowGi + i);
            }
        }

        for (; i < d; i++)
        {
            float dy = rowGo[i];
            float xHat = (rowIn[i] - mean) * invStd;
            float g = pG != null ? dy * pG[i] : dy;
            rowGi[i] = invStd * (g - c1 - xHat * c2);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RmsNormRowSimd(float* rowIn, float* pW, float* rowOut, int d, float eps)
    {
        double sumSq = SimdMath.WidenSumSquares(rowIn, d);
        float invRms = 1.0f / MathF.Sqrt((float)(sumSq / d) + eps);

        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vInvRms = Vector512.Create(invRms);
            int step = Vector512<float>.Count;
            for (; i + step <= d; i += step)
            {
                var scaled = Vector512.Load(rowIn + i) * vInvRms;
                if (pW != null) scaled *= Vector512.Load(pW + i);
                Vector512.Store(scaled, rowOut + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vInvRms = Vector256.Create(invRms);
            int step = Vector256<float>.Count;
            for (; i + step <= d; i += step)
            {
                var scaled = Vector256.Load(rowIn + i) * vInvRms;
                if (pW != null) scaled *= Vector256.Load(pW + i);
                Vector256.Store(scaled, rowOut + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vInvRms = Vector128.Create(invRms);
            int step = Vector128<float>.Count;
            for (; i + step <= d; i += step)
            {
                var scaled = Vector128.Load(rowIn + i) * vInvRms;
                if (pW != null) scaled *= Vector128.Load(pW + i);
                Vector128.Store(scaled, rowOut + i);
            }
        }

        for (; i < d; i++)
        {
            float scaled = rowIn[i] * invRms;
            rowOut[i] = pW != null ? scaled * pW[i] : scaled;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RmsNormBackwardRowSimd(
        float* rowGo,
        float* rowIn,
        float* pW,
        float* rowGi,
        float* pGw,
        int d,
        float eps)
    {
        double sumSq = SimdMath.WidenSumSquares(rowIn, d);
        float invRms = 1.0f / MathF.Sqrt((float)(sumSq / d) + eps);

        double dotGAndXHat = 0.0;
        int i = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var vInvRms = Vector512.Create(invRms);
            var accLo = Vector512<double>.Zero;
            var accHi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector512.Load(rowGo + i);
                var xHat = Vector512.Load(rowIn + i) * vInvRms;
                var g = pW != null ? dy * Vector512.Load(pW + i) : dy;
                var (lo, hi) = Vector512.Widen(g * xHat);
                accLo += lo;
                accHi += hi;
                if (pGw != null) Vector512.Store(Vector512.Load(pGw + i) + dy * xHat, pGw + i);
            }
            dotGAndXHat += Vector512.Sum(accLo + accHi);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vInvRms = Vector256.Create(invRms);
            var accLo = Vector256<double>.Zero;
            var accHi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector256.Load(rowGo + i);
                var xHat = Vector256.Load(rowIn + i) * vInvRms;
                var g = pW != null ? dy * Vector256.Load(pW + i) : dy;
                var (lo, hi) = Vector256.Widen(g * xHat);
                accLo += lo;
                accHi += hi;
                if (pGw != null) Vector256.Store(Vector256.Load(pGw + i) + dy * xHat, pGw + i);
            }
            dotGAndXHat += Vector256.Sum(accLo + accHi);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vInvRms = Vector128.Create(invRms);
            var accLo = Vector128<double>.Zero;
            var accHi = Vector128<double>.Zero;
            int step = Vector128<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector128.Load(rowGo + i);
                var xHat = Vector128.Load(rowIn + i) * vInvRms;
                var g = pW != null ? dy * Vector128.Load(pW + i) : dy;
                var (lo, hi) = Vector128.Widen(g * xHat);
                accLo += lo;
                accHi += hi;
                if (pGw != null) Vector128.Store(Vector128.Load(pGw + i) + dy * xHat, pGw + i);
            }
            dotGAndXHat += Vector128.Sum(accLo + accHi);
        }

        for (; i < d; i++)
        {
            float dy = rowGo[i];
            float xHat = rowIn[i] * invRms;
            float g = pW != null ? dy * pW[i] : dy;
            dotGAndXHat += g * xHat;
            if (pGw != null) pGw[i] += dy * xHat;
        }

        float scale = (float)(dotGAndXHat / d);
        i = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var vInvRms = Vector512.Create(invRms);
            var vScale = Vector512.Create(scale);
            int step = Vector512<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector512.Load(rowGo + i);
                var xHat = Vector512.Load(rowIn + i) * vInvRms;
                var g = pW != null ? dy * Vector512.Load(pW + i) : dy;
                Vector512.Store(vInvRms * (g - xHat * vScale), rowGi + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vInvRms = Vector256.Create(invRms);
            var vScale = Vector256.Create(scale);
            int step = Vector256<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector256.Load(rowGo + i);
                var xHat = Vector256.Load(rowIn + i) * vInvRms;
                var g = pW != null ? dy * Vector256.Load(pW + i) : dy;
                Vector256.Store(vInvRms * (g - xHat * vScale), rowGi + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            var vInvRms = Vector128.Create(invRms);
            var vScale = Vector128.Create(scale);
            int step = Vector128<float>.Count;
            for (; i + step <= d; i += step)
            {
                var dy = Vector128.Load(rowGo + i);
                var xHat = Vector128.Load(rowIn + i) * vInvRms;
                var g = pW != null ? dy * Vector128.Load(pW + i) : dy;
                Vector128.Store(vInvRms * (g - xHat * vScale), rowGi + i);
            }
        }

        for (; i < d; i++)
        {
            float dy = rowGo[i];
            float xHat = rowIn[i] * invRms;
            float g = pW != null ? dy * pW[i] : dy;
            rowGi[i] = invRms * (g - xHat * scale);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float FindMaxSimd(float* data, int n)
    {
        float maxVal = float.NegativeInfinity;
        int i = 0;

        if (Vector512.IsHardwareAccelerated && n >= Vector512<float>.Count)
        {
            var vMax = Vector512.Load(data);
            int step = Vector512<float>.Count;
            i = step;
            for (; i + step <= n; i += step)
            {
                vMax = Vector512.Max(vMax, Vector512.Load(data + i));
            }
            for (int lane = 0; lane < step; lane++)
            {
                if (vMax[lane] > maxVal) maxVal = vMax[lane];
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= Vector256<float>.Count)
        {
            var vMax = Vector256.Load(data);
            int step = Vector256<float>.Count;
            i = step;
            for (; i + step <= n; i += step)
            {
                vMax = Vector256.Max(vMax, Vector256.Load(data + i));
            }
            for (int lane = 0; lane < step; lane++)
            {
                if (vMax[lane] > maxVal) maxVal = vMax[lane];
            }
        }
        else if (Vector128.IsHardwareAccelerated && n >= Vector128<float>.Count)
        {
            var vMax = Vector128.Load(data);
            int step = Vector128<float>.Count;
            i = step;
            for (; i + step <= n; i += step)
            {
                vMax = Vector128.Max(vMax, Vector128.Load(data + i));
            }
            for (int lane = 0; lane < step; lane++)
            {
                if (vMax[lane] > maxVal) maxVal = vMax[lane];
            }
        }

        for (; i < n; i++)
        {
            if (data[i] > maxVal) maxVal = data[i];
        }

        return maxVal;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double SumSquaredDiffSimd(float* data, float mean, int n)
    {
        double total = 0.0;
        int i = 0;

        if (Vector512.IsHardwareAccelerated && n >= Vector512<float>.Count)
        {
            var vMean = Vector512.Create(mean);
            var accLo = Vector512<double>.Zero;
            var accHi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                var diff = Vector512.Load(data + i) - vMean;
                var (lo, hi) = Vector512.Widen(diff * diff);
                accLo += lo;
                accHi += hi;
            }
            total += Vector512.Sum(accLo + accHi);
        }
        else if (Vector256.IsHardwareAccelerated && n >= Vector256<float>.Count)
        {
            var vMean = Vector256.Create(mean);
            var accLo = Vector256<double>.Zero;
            var accHi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                var diff = Vector256.Load(data + i) - vMean;
                var (lo, hi) = Vector256.Widen(diff * diff);
                accLo += lo;
                accHi += hi;
            }
            total += Vector256.Sum(accLo + accHi);
        }

        if (Vector128.IsHardwareAccelerated && i + Vector128<float>.Count <= n)
        {
            var vMean = Vector128.Create(mean);
            var accLo = Vector128<double>.Zero;
            var accHi = Vector128<double>.Zero;
            int step = Vector128<float>.Count;
            for (; i + step <= n; i += step)
            {
                var diff = Vector128.Load(data + i) - vMean;
                var (lo, hi) = Vector128.Widen(diff * diff);
                accLo += lo;
                accHi += hi;
            }
            total += Vector128.Sum(accLo + accHi);
        }

        for (; i < n; i++)
        {
            double diff = data[i] - mean;
            total += diff * diff;
        }

        return total;
    }
}
