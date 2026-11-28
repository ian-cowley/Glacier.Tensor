using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Losses;

/// <summary>
/// Differentiable loss functions for deep learning training.
/// Supports both <c>tape.Backward(lossTensor)</c> and legacy <c>tape.Backward(predictions)</c>.
/// </summary>
public static unsafe class LossFunctions
{
    public static (float lossValue, Tensor<float> lossTensor) MSELoss(Tensor<float> predictions, Tensor<float> targets)
    {
        ArgumentNullException.ThrowIfNull(predictions);
        ArgumentNullException.ThrowIfNull(targets);
        if (predictions.ElementCount != targets.ElementCount)
            throw new ArgumentException("Prediction and target shapes must match.");

        int n = checked((int)predictions.ElementCount);
        if (n == 0)
            throw new ArgumentException("Loss requires at least one element.");

        using var pContig = (predictions.IsContiguous && predictions.ElementOffset == 0) ? null : predictions.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        var pSpan = (pContig ?? predictions).AsSpan();
        var tSpan = (tContig ?? targets).AsSpan();

        double totalLoss = 0.0;
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var accLo = Vector512<double>.Zero;
            var accHi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            fixed (float* pp = pSpan, pt = tSpan)
            {
                for (; i + step <= n; i += step)
                {
                    var d = Vector512.Load(pp + i) - Vector512.Load(pt + i);
                    var (lo, hi) = Vector512.Widen(d * d);
                    accLo += lo;
                    accHi += hi;
                }
            }
            totalLoss += Vector512.Sum(accLo + accHi);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var accLo = Vector256<double>.Zero;
            var accHi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            fixed (float* pp = pSpan, pt = tSpan)
            {
                for (; i + step <= n; i += step)
                {
                    var d = Vector256.Load(pp + i) - Vector256.Load(pt + i);
                    var (lo, hi) = Vector256.Widen(d * d);
                    accLo += lo;
                    accHi += hi;
                }
            }
            totalLoss += Vector256.Sum(accLo + accHi);
        }

        for (; i < n; i++)
        {
            double d = (double)pSpan[i] - tSpan[i];
            totalLoss += d * d;
        }

        float meanLoss = (float)(totalLoss / n);
        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = meanLoss;
        lossTensor.RequiresGrad = predictions.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape && lossTensor.RequiresGrad)
        {
            tape.Record(new TapeEntry(AutogradOp.MSELoss, lossTensor, predictions, targets));
        }

        return (meanLoss, lossTensor);
    }

    internal static void ComputeMSEGradient(
        Tensor<float> predictions,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss,
        bool accumulate = false)
    {
        int n = checked((int)predictions.ElementCount);
        using var pContig = (predictions.IsContiguous && predictions.ElementOffset == 0) ? null : predictions.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        using var gContig = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0)
            ? null
            : (accumulate ? gradOutput.Contiguous() : new Tensor<float>(gradOutput.Shape));

        var pSrc = pContig ?? predictions;
        var tSrc = tContig ?? targets;
        var gDst = gContig ?? gradOutput;

        float* pp = pSrc.DataPointer;
        float* pt = tSrc.DataPointer;
        float* pg = gDst.DataPointer;

        float scale = (2.0f * dLoss) / n;
        int i = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            var vScale = Vector512.Create(scale);
            int step = Vector512<float>.Count;
            for (; i + step <= n; i += step)
            {
                var grad = vScale * (Vector512.Load(pp + i) - Vector512.Load(pt + i));
                if (accumulate) grad += Vector512.Load(pg + i);
                Vector512.Store(grad, pg + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vScale = Vector256.Create(scale);
            int step = Vector256<float>.Count;
            for (; i + step <= n; i += step)
            {
                var grad = vScale * (Vector256.Load(pp + i) - Vector256.Load(pt + i));
                if (accumulate) grad += Vector256.Load(pg + i);
                Vector256.Store(grad, pg + i);
            }
        }

        for (; i < n; i++)
        {
            float grad = scale * (pp[i] - pt[i]);
            pg[i] = accumulate ? pg[i] + grad : grad;
        }

        if (gContig != null)
        {
            Tensor<float>.CopyElements(gContig, gradOutput);
        }
    }

    public static (float lossValue, Tensor<float> lossTensor) BinaryCrossEntropy(Tensor<float> predictions, Tensor<float> targets)
    {
        ArgumentNullException.ThrowIfNull(predictions);
        ArgumentNullException.ThrowIfNull(targets);
        if (predictions.ElementCount != targets.ElementCount)
            throw new ArgumentException("Prediction and target shapes must match.");

        int n = checked((int)predictions.ElementCount);
        if (n == 0)
            throw new ArgumentException("Loss requires at least one element.");

        using var pContig = (predictions.IsContiguous && predictions.ElementOffset == 0) ? null : predictions.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        var pSpan = (pContig ?? predictions).AsSpan();
        var tSpan = (tContig ?? targets).AsSpan();

        double totalLoss = 0.0;
        const float eps = 1e-7f;

        for (int i = 0; i < n; i++)
        {
            float p = Math.Clamp(pSpan[i], eps, 1.0f - eps);
            float y = tSpan[i];
            totalLoss -= (double)y * MathF.Log(p) + (1.0 - y) * MathF.Log(1.0f - p);
        }

        float meanLoss = (float)(totalLoss / n);
        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = meanLoss;
        lossTensor.RequiresGrad = predictions.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape && lossTensor.RequiresGrad)
        {
            tape.Record(new TapeEntry(AutogradOp.BinaryCrossEntropyLoss, lossTensor, predictions, targets));
        }

        return (meanLoss, lossTensor);
    }

    internal static void ComputeBinaryCrossEntropyGradient(
        Tensor<float> predictions,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss,
        bool accumulate = false)
    {
        const float eps = 1e-7f;
        int n = checked((int)predictions.ElementCount);
        using var pContig = (predictions.IsContiguous && predictions.ElementOffset == 0) ? null : predictions.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        using var gContig = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0)
            ? null
            : (accumulate ? gradOutput.Contiguous() : new Tensor<float>(gradOutput.Shape));

        var pSpan = (pContig ?? predictions).AsSpan();
        var tSpan = (tContig ?? targets).AsSpan();
        var gSpan = (gContig ?? gradOutput).AsSpan();

        float scale = dLoss / n;
        if (accumulate)
        {
            for (int i = 0; i < n; i++)
            {
                float p = Math.Clamp(pSpan[i], eps, 1.0f - eps);
                float y = tSpan[i];
                gSpan[i] += scale * (p - y) / (p * (1.0f - p));
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                float p = Math.Clamp(pSpan[i], eps, 1.0f - eps);
                float y = tSpan[i];
                gSpan[i] = scale * (p - y) / (p * (1.0f - p));
            }
        }

        if (gContig != null)
        {
            Tensor<float>.CopyElements(gContig, gradOutput);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Log1p(double x) => FusedCrossEntropyLoss.Log1p(x);

    private static void ComputeSplitSumSoftmaxStats(
        float* rowLogits,
        int classes,
        out double maxLogit,
        out int argMax,
        out double sumOther)
    {
        argMax = 0;
        maxLogit = rowLogits[0];
        for (int c = 1; c < classes; c++)
        {
            float v = rowLogits[c];
            if (v > maxLogit)
            {
                maxLogit = v;
                argMax = c;
            }
        }

        sumOther = 0.0;
        for (int c = 0; c < classes; c++)
        {
            if (c != argMax)
            {
                sumOther += Math.Exp((double)rowLogits[c] - maxLogit);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ResolveBatchClasses(Tensor<float> logits, out int batch, out int classes)
    {
        ArgumentNullException.ThrowIfNull(logits);
        if (logits.Rank == 0)
            throw new ArgumentException("Logits tensor must have Rank >= 1.");

        classes = logits.Rank > 1 ? logits.Shape[logits.Rank - 1] : logits.Shape[0];
        batch = classes > 0 ? checked((int)(logits.ElementCount / classes)) : 0;

        if (batch <= 0 || classes <= 0)
            throw new ArgumentException("Invalid logits tensor dimensions.");
    }

    /// <summary>
    /// Computes multi-class categorical cross-entropy loss with fused numerical-stable softmax.
    /// Supports Rank >= 1 logits (<c>[..., C]</c>) with one-hot targets or class index targets.
    /// Automatically records on AutogradTape for reverse-mode automatic differentiation.
    /// </summary>
    public static (float lossValue, Tensor<float> lossTensor) CrossEntropy(Tensor<float> logits, Tensor<float> targets)
    {
        ArgumentNullException.ThrowIfNull(targets);
        ResolveBatchClasses(logits, out int batch, out int classes);

        using var lContig = (logits.IsContiguous && logits.ElementOffset == 0) ? null : logits.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        var lSrc = lContig ?? logits;
        var tSrc = tContig ?? targets;

        float* pLog = lSrc.DataPointer;
        float* pTgt = tSrc.DataPointer;
        bool isOneHot = targets.ElementCount == (long)batch * classes;

        double totalLoss = 0.0;

        for (int b = 0; b < batch; b++)
        {
            long offset = (long)b * classes;
            float* rowLogits = pLog + offset;
            ComputeSplitSumSoftmaxStats(rowLogits, classes, out double maxLogit, out int argMax, out double sumOther);
            double logZOffset = Log1p(sumOther);

            if (isOneHot)
            {
                for (int c = 0; c < classes; c++)
                {
                    double y = pTgt[offset + c];
                    if (y > 0.0)
                    {
                        double negLogP = (c == argMax)
                            ? logZOffset
                            : (logZOffset + (maxLogit - (double)rowLogits[c]));
                        totalLoss += y * negLogP;
                    }
                }
            }
            else
            {
                int targetClass = (int)pTgt[b];
                if (targetClass >= 0 && targetClass < classes)
                {
                    double negLogP = (targetClass == argMax)
                        ? logZOffset
                        : (logZOffset + (maxLogit - (double)rowLogits[targetClass]));
                    totalLoss += negLogP;
                }
            }
        }

        float meanLoss = (float)(totalLoss / batch);

        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = meanLoss;
        lossTensor.RequiresGrad = logits.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape && lossTensor.RequiresGrad)
        {
            tape.Record(new TapeEntry(AutogradOp.CrossEntropyLoss, lossTensor, logits, targets));
        }

        return (meanLoss, lossTensor);
    }

    internal static void ComputeCrossEntropyGradient(
        Tensor<float> logits,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss)
    {
        ResolveBatchClasses(logits, out int batch, out int classes);

        using var lContig = (logits.IsContiguous && logits.ElementOffset == 0) ? null : logits.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        using var gContig = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0) ? null : new Tensor<float>(gradOutput.Shape);

        var lSrc = lContig ?? logits;
        var tSrc = tContig ?? targets;
        var gDst = gContig ?? gradOutput;

        float* pLog = lSrc.DataPointer;
        float* pTgt = tSrc.DataPointer;
        float* pOut = gDst.DataPointer;
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        double dScale = (double)dLoss / batch;

        for (int b = 0; b < batch; b++)
        {
            long offset = (long)b * classes;
            float* rowLogits = pLog + offset;
            ComputeSplitSumSoftmaxStats(rowLogits, classes, out double maxLogit, out int argMax, out double sumOther);
            double invDenom = 1.0 / (1.0 + sumOther);

            int targetClass = isOneHot ? -1 : (int)pTgt[b];

            for (int c = 0; c < classes; c++)
            {
                double y = isOneHot ? pTgt[offset + c] : (targetClass == c ? 1.0 : 0.0);
                double diff = (c == argMax)
                    ? (((1.0 - y) - y * sumOther) * invDenom)
                    : (Math.Exp((double)rowLogits[c] - maxLogit) * invDenom - y);
                pOut[offset + c] = (float)(dScale * diff);
            }
        }

        if (gContig != null)
        {
            Tensor<float>.CopyElements(gContig, gradOutput);
        }
    }

    /// <summary>
    /// Computes multi-class categorical cross-entropy loss with optional label smoothing.
    /// Smooths one-hot labels with uniform prior: y'_k = (1 - alpha) * y_k + alpha / K.
    /// </summary>
    public static (float lossValue, Tensor<float> lossTensor) CategoricalCrossEntropyLoss(
        Tensor<float> logits,
        Tensor<float> targets,
        float labelSmoothing = 0.0f)
    {
        ArgumentNullException.ThrowIfNull(targets);
        ResolveBatchClasses(logits, out int batch, out int classes);

        using var lContig = (logits.IsContiguous && logits.ElementOffset == 0) ? null : logits.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        var lSrc = lContig ?? logits;
        var tSrc = tContig ?? targets;

        float* pLog = lSrc.DataPointer;
        float* pTgt = tSrc.DataPointer;
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        double totalLoss = 0.0;
        double smoothing = labelSmoothing;
        double smoothVal = smoothing / classes;

        for (int b = 0; b < batch; b++)
        {
            long offset = (long)b * classes;
            float* rowLogits = pLog + offset;
            ComputeSplitSumSoftmaxStats(rowLogits, classes, out double maxLogit, out int argMax, out double sumOther);
            double logZOffset = Log1p(sumOther);

            int targetClass = isOneHot ? -1 : (int)pTgt[b];
            for (int c = 0; c < classes; c++)
            {
                double y = isOneHot ? pTgt[offset + c] : (targetClass == c ? 1.0 : 0.0);
                double smoothedY = (1.0 - smoothing) * y + smoothVal;
                double negLogP = (c == argMax)
                    ? logZOffset
                    : (logZOffset + (maxLogit - (double)rowLogits[c]));
                totalLoss += smoothedY * negLogP;
            }
        }

        float meanLoss = (float)(totalLoss / batch);
        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = meanLoss;
        lossTensor.RequiresGrad = logits.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape && lossTensor.RequiresGrad)
        {
            tape.Record(new TapeEntry(AutogradOp.CategoricalCrossEntropyLoss, lossTensor, logits, targets, labelSmoothing));
        }

        return (meanLoss, lossTensor);
    }

    internal static void ComputeCategoricalCrossEntropyGradient(
        Tensor<float> logits,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float labelSmoothing,
        float dLoss,
        bool accumulate = false)
    {
        ResolveBatchClasses(logits, out int batch, out int classes);

        using var lContig = (logits.IsContiguous && logits.ElementOffset == 0) ? null : logits.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        using var gContig = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0)
            ? null
            : (accumulate ? gradOutput.Contiguous() : new Tensor<float>(gradOutput.Shape));

        var lSrc = lContig ?? logits;
        var tSrc = tContig ?? targets;
        var gDst = gContig ?? gradOutput;

        float* pLog = lSrc.DataPointer;
        float* pTgt = tSrc.DataPointer;
        float* pOut = gDst.DataPointer;
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        double smoothing = labelSmoothing;
        double smoothVal = smoothing / classes;
        double dScale = (double)dLoss / batch;

        for (int b = 0; b < batch; b++)
        {
            long offset = (long)b * classes;
            float* rowLogits = pLog + offset;
            ComputeSplitSumSoftmaxStats(rowLogits, classes, out double maxLogit, out int argMax, out double sumOther);
            double invDenom = 1.0 / (1.0 + sumOther);

            int targetClass = isOneHot ? -1 : (int)pTgt[b];
            for (int c = 0; c < classes; c++)
            {
                double y = isOneHot ? pTgt[offset + c] : (targetClass == c ? 1.0 : 0.0);
                double smoothedY = (1.0 - smoothing) * y + smoothVal;
                double diff = (c == argMax)
                    ? (((1.0 - smoothedY) - smoothedY * sumOther) * invDenom)
                    : (Math.Exp((double)rowLogits[c] - maxLogit) * invDenom - smoothedY);
                float grad = (float)(dScale * diff);
                if (accumulate)
                    pOut[offset + c] += grad;
                else
                    pOut[offset + c] = grad;
            }
        }

        if (gContig != null)
        {
            Tensor<float>.CopyElements(gContig, gradOutput);
        }
    }

    /// <summary>
    /// Computes Brier Score Loss (mean squared error of predicted probabilities against ground truth).
    /// Strictly proper scoring rule used for policy probability calibration.
    /// </summary>
    public static (float lossValue, Tensor<float> lossTensor) BrierScoreLoss(
        Tensor<float> logits,
        Tensor<float> targets)
    {
        ArgumentNullException.ThrowIfNull(targets);
        ResolveBatchClasses(logits, out int batch, out int classes);

        using var lContig = (logits.IsContiguous && logits.ElementOffset == 0) ? null : logits.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        var lSrc = lContig ?? logits;
        var tSrc = tContig ?? targets;

        float* pLog = lSrc.DataPointer;
        float* pTgt = tSrc.DataPointer;
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        double totalLoss = 0.0;

        float[]? rented = classes > 256 ? ArrayPool<float>.Shared.Rent(classes) : null;
        try
        {
            Span<float> pSpan = rented != null ? rented.AsSpan(0, classes) : stackalloc float[classes];

            for (int b = 0; b < batch; b++)
            {
                long offset = (long)b * classes;
                float* rowLogits = pLog + offset;

                float maxLogit = rowLogits[0];
                for (int c = 1; c < classes; c++)
                {
                    if (rowLogits[c] > maxLogit) maxLogit = rowLogits[c];
                }

                double sumExp = 0.0;
                for (int c = 0; c < classes; c++)
                {
                    float e = SimdMath.FastExpNonPositiveScalar(rowLogits[c] - maxLogit);
                    pSpan[c] = e;
                    sumExp += e;
                }

                float invSum = (float)(1.0 / sumExp);
                int targetClass = isOneHot ? -1 : (int)pTgt[b];
                for (int c = 0; c < classes; c++)
                {
                    pSpan[c] *= invSum;
                    float y = isOneHot ? pTgt[offset + c] : (targetClass == c ? 1.0f : 0.0f);
                    double diff = (double)pSpan[c] - y;
                    totalLoss += diff * diff;
                }
            }
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        float meanLoss = (float)(totalLoss / batch);
        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = meanLoss;
        lossTensor.RequiresGrad = logits.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape && lossTensor.RequiresGrad)
        {
            tape.Record(new TapeEntry(AutogradOp.BrierScoreLoss, lossTensor, logits, targets));
        }

        return (meanLoss, lossTensor);
    }

    internal static void ComputeBrierScoreGradient(
        Tensor<float> logits,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss,
        bool accumulate = false)
    {
        ResolveBatchClasses(logits, out int batch, out int classes);

        using var lContig = (logits.IsContiguous && logits.ElementOffset == 0) ? null : logits.Contiguous();
        using var tContig = (targets.IsContiguous && targets.ElementOffset == 0) ? null : targets.Contiguous();
        using var gContig = (gradOutput.IsContiguous && gradOutput.ElementOffset == 0)
            ? null
            : (accumulate ? gradOutput.Contiguous() : new Tensor<float>(gradOutput.Shape));

        var lSrc = lContig ?? logits;
        var tSrc = tContig ?? targets;
        var gDst = gContig ?? gradOutput;

        float* pLog = lSrc.DataPointer;
        float* pTgt = tSrc.DataPointer;
        float* pOut = gDst.DataPointer;
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        float scale = (2.0f * dLoss) / batch;

        float[]? rented = classes > 256 ? ArrayPool<float>.Shared.Rent(classes) : null;
        try
        {
            Span<float> pSpan = rented != null ? rented.AsSpan(0, classes) : stackalloc float[classes];

            for (int b = 0; b < batch; b++)
            {
                long offset = (long)b * classes;
                float* rowLogits = pLog + offset;

                float maxLogit = rowLogits[0];
                for (int c = 1; c < classes; c++)
                {
                    if (rowLogits[c] > maxLogit) maxLogit = rowLogits[c];
                }

                double sumExp = 0.0;
                for (int c = 0; c < classes; c++)
                {
                    float e = SimdMath.FastExpNonPositiveScalar(rowLogits[c] - maxLogit);
                    pSpan[c] = e;
                    sumExp += e;
                }

                float invSum = (float)(1.0 / sumExp);
                int targetClass = isOneHot ? -1 : (int)pTgt[b];
                double dotDiffProb = 0.0;

                for (int c = 0; c < classes; c++)
                {
                    pSpan[c] *= invSum;
                    float y = isOneHot ? pTgt[offset + c] : (targetClass == c ? 1.0f : 0.0f);
                    dotDiffProb += ((double)pSpan[c] - y) * pSpan[c];
                }

                float dotF = (float)dotDiffProb;
                for (int c = 0; c < classes; c++)
                {
                    float y = isOneHot ? pTgt[offset + c] : (targetClass == c ? 1.0f : 0.0f);
                    float grad = scale * pSpan[c] * ((pSpan[c] - y) - dotF);
                    if (accumulate)
                        pOut[offset + c] += grad;
                    else
                        pOut[offset + c] = grad;
                }
            }
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        if (gContig != null)
        {
            Tensor<float>.CopyElements(gContig, gradOutput);
        }
    }

    /// <summary>
    /// Computes Expected Calibration Error (ECE) across M confidence bins.
    /// Measures the gap between predicted confidence and empirical accuracy.
    /// </summary>
    public static float ExpectedCalibrationError(
        ReadOnlySpan<float> confidences,
        ReadOnlySpan<int> predictions,
        ReadOnlySpan<int> groundTruth,
        int numBins = 10)
    {
        if (numBins <= 0)
            throw new ArgumentOutOfRangeException(nameof(numBins), "Number of bins must be positive.");
        if (confidences.Length != predictions.Length || confidences.Length != groundTruth.Length)
            throw new ArgumentException("Input spans must have the same length.");

        int n = confidences.Length;
        if (n == 0) return 0f;

        bool useHeap = numBins > 256;
        Span<int> binCounts = useHeap ? new int[numBins] : stackalloc int[numBins];
        Span<float> binAccuracies = useHeap ? new float[numBins] : stackalloc float[numBins];
        Span<float> binConfidences = useHeap ? new float[numBins] : stackalloc float[numBins];

        for (int i = 0; i < n; i++)
        {
            float conf = Math.Clamp(confidences[i], 0f, 1f);
            int bin = Math.Min((int)(conf * numBins), numBins - 1);
            binCounts[bin]++;
            binConfidences[bin] += conf;
            if (predictions[i] == groundTruth[i])
            {
                binAccuracies[bin] += 1.0f;
            }
        }

        float ece = 0f;
        for (int b = 0; b < numBins; b++)
        {
            if (binCounts[b] > 0)
            {
                float avgAcc = binAccuracies[b] / binCounts[b];
                float avgConf = binConfidences[b] / binCounts[b];
                ece += (float)binCounts[b] / n * MathF.Abs(avgAcc - avgConf);
            }
        }

        return ece;
    }
}
