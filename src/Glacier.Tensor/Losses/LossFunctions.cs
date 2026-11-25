using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Losses;

/// <summary>
/// Differentiable loss functions for deep learning training.
/// Supports both <c>tape.Backward(lossTensor)</c> and direct <c>tape.Backward(predictions)</c>.
/// </summary>
public static class LossFunctions
{
    public static (float lossValue, Tensor<float> lossTensor) MSELoss(Tensor<float> predictions, Tensor<float> targets)
    {
        if (predictions.ElementCount != targets.ElementCount)
            throw new ArgumentException("Prediction and target shapes must match.");

        int n = (int)predictions.ElementCount;
        using var pContig = predictions.IsContiguous ? null : predictions.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var pSpan = (pContig ?? predictions).AsSpan();
        var tSpan = (tContig ?? targets).AsSpan();

        float totalLoss = 0f;
        for (int i = 0; i < n; i++)
        {
            float d = pSpan[i] - tSpan[i];
            totalLoss += d * d;
        }
        totalLoss /= n;

        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = totalLoss;
        lossTensor.RequiresGrad = predictions.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape)
        {
            if (predictions.Grad == null)
                predictions.Grad = Tensor<float>.Zeros(predictions.Shape);

            ComputeMSEGradient(predictions, targets, predictions.Grad, 1.0f, accumulate: true);
            tape.Record(new TapeEntry(AutogradOp.MSELoss, lossTensor, predictions, targets));
        }

        return (totalLoss, lossTensor);
    }

    internal static void ComputeMSEGradient(
        Tensor<float> predictions,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss,
        bool accumulate = false)
    {
        int n = (int)predictions.ElementCount;
        using var pContig = predictions.IsContiguous ? null : predictions.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var pSpan = (pContig ?? predictions).AsSpan();
        var tSpan = (tContig ?? targets).AsSpan();
        var gSpan = gradOutput.AsSpan();

        float scale = (2.0f * dLoss) / n;
        if (accumulate)
        {
            for (int i = 0; i < n; i++)
            {
                gSpan[i] += scale * (pSpan[i] - tSpan[i]);
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                gSpan[i] = scale * (pSpan[i] - tSpan[i]);
            }
        }
    }

    public static (float lossValue, Tensor<float> lossTensor) BinaryCrossEntropy(Tensor<float> predictions, Tensor<float> targets)
    {
        if (predictions.ElementCount != targets.ElementCount)
            throw new ArgumentException("Prediction and target shapes must match.");

        int n = (int)predictions.ElementCount;
        using var pContig = predictions.IsContiguous ? null : predictions.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var pSpan = (pContig ?? predictions).AsSpan();
        var tSpan = (tContig ?? targets).AsSpan();

        float totalLoss = 0f;
        const float eps = 1e-7f;

        for (int i = 0; i < n; i++)
        {
            float p = Math.Clamp(pSpan[i], eps, 1.0f - eps);
            float y = tSpan[i];
            totalLoss -= y * MathF.Log(p) + (1.0f - y) * MathF.Log(1.0f - p);
        }
        totalLoss /= n;

        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = totalLoss;
        lossTensor.RequiresGrad = predictions.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape)
        {
            if (predictions.Grad == null)
                predictions.Grad = Tensor<float>.Zeros(predictions.Shape);

            ComputeBinaryCrossEntropyGradient(predictions, targets, predictions.Grad, 1.0f, accumulate: true);
            tape.Record(new TapeEntry(AutogradOp.BinaryCrossEntropyLoss, lossTensor, predictions, targets));
        }

        return (totalLoss, lossTensor);
    }

    internal static void ComputeBinaryCrossEntropyGradient(
        Tensor<float> predictions,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss,
        bool accumulate = false)
    {
        const float eps = 1e-7f;
        int n = (int)predictions.ElementCount;
        using var pContig = predictions.IsContiguous ? null : predictions.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var pSpan = (pContig ?? predictions).AsSpan();
        var tSpan = (tContig ?? targets).AsSpan();
        var gSpan = gradOutput.AsSpan();

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
    }

    private static double Log1p(double x)
    {
        if (x == 0.0) return 0.0;
        double u = 1.0 + x;
        if (u == 1.0) return x;
        return Math.Log(u) * (x / (u - 1.0));
    }

    private static void ComputeSplitSumSoftmaxStats(
        ReadOnlySpan<float> rowLogits,
        out double maxLogit,
        out int argMax,
        out double sumOther)
    {
        argMax = 0;
        maxLogit = rowLogits[0];
        for (int c = 1; c < rowLogits.Length; c++)
        {
            float v = rowLogits[c];
            if (v > maxLogit)
            {
                maxLogit = v;
                argMax = c;
            }
        }

        sumOther = 0.0;
        for (int c = 0; c < rowLogits.Length; c++)
        {
            if (c != argMax)
                sumOther += Math.Exp((double)rowLogits[c] - maxLogit);
        }
    }

    /// <summary>
    /// Computes multi-class categorical cross-entropy loss with fused numerical-stable softmax.
    /// Supports one-hot targets of shape [B, C] or class index targets of shape [B] or [B, 1].
    /// Automatically records on AutogradTape for reverse-mode automatic differentiation.
    /// </summary>
    public static (float lossValue, Tensor<float> lossTensor) CrossEntropy(Tensor<float> logits, Tensor<float> targets)
    {
        int batch = logits.Shape.Rank > 1 ? logits.Shape[0] : 1;
        int classes = logits.Shape.Rank > 1 ? logits.Shape[1] : logits.Shape[0];

        if (batch <= 0 || classes <= 0)
            throw new ArgumentException("Invalid logits tensor dimensions.");

        using var lContig = logits.IsContiguous ? null : logits.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var logSpan = (lContig ?? logits).AsSpan();
        var tgtSpan = (tContig ?? targets).AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;

        double totalLoss = 0.0;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;
            ReadOnlySpan<float> rowLogits = logSpan.Slice(offset, classes);
            ComputeSplitSumSoftmaxStats(rowLogits, out double maxLogit, out int argMax, out double sumOther);
            double logZOffset = Log1p(sumOther);

            if (isOneHot)
            {
                for (int c = 0; c < classes; c++)
                {
                    double y = tgtSpan[offset + c];
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
                int targetClass = (int)tgtSpan[b];
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

        if (AutogradTape.Current is { IsExecutingBackward: false } tape)
        {
            if (logits.Grad == null)
                logits.Grad = Tensor<float>.Zeros(logits.Shape);

            ComputeCrossEntropyGradient(logits, targets, logits.Grad, 1.0f);
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
        int batch = logits.Shape.Rank > 1 ? logits.Shape[0] : 1;
        int classes = logits.Shape.Rank > 1 ? logits.Shape[1] : logits.Shape[0];

        using var lContig = logits.IsContiguous ? null : logits.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var logSpan = (lContig ?? logits).AsSpan();
        var tgtSpan = (tContig ?? targets).AsSpan();
        var outSpan = gradOutput.AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        double dScale = (double)dLoss / batch;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;
            ReadOnlySpan<float> rowLogits = logSpan.Slice(offset, classes);
            ComputeSplitSumSoftmaxStats(rowLogits, out double maxLogit, out int argMax, out double sumOther);
            double invDenom = 1.0 / (1.0 + sumOther);

            int targetClass = isOneHot ? -1 : (int)tgtSpan[b];

            for (int c = 0; c < classes; c++)
            {
                double y = isOneHot ? tgtSpan[offset + c] : (targetClass == c ? 1.0 : 0.0);
                double diff = (c == argMax)
                    ? (((1.0 - y) - y * sumOther) * invDenom)
                    : (Math.Exp((double)rowLogits[c] - maxLogit) * invDenom - y);
                outSpan[offset + c] = (float)(dScale * diff);
            }
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
        int batch = logits.Shape.Rank > 1 ? logits.Shape[0] : 1;
        int classes = logits.Shape.Rank > 1 ? logits.Shape[1] : logits.Shape[0];

        if (batch <= 0 || classes <= 0)
            throw new ArgumentException("Invalid logits tensor dimensions.");

        using var lContig = logits.IsContiguous ? null : logits.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var logSpan = (lContig ?? logits).AsSpan();
        var tgtSpan = (tContig ?? targets).AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        double totalLoss = 0.0;
        double smoothing = labelSmoothing;
        double smoothVal = smoothing / classes;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;
            ReadOnlySpan<float> rowLogits = logSpan.Slice(offset, classes);
            ComputeSplitSumSoftmaxStats(rowLogits, out double maxLogit, out int argMax, out double sumOther);
            double logZOffset = Log1p(sumOther);

            int targetClass = isOneHot ? -1 : (int)tgtSpan[b];
            for (int c = 0; c < classes; c++)
            {
                double y = isOneHot ? tgtSpan[offset + c] : (targetClass == c ? 1.0 : 0.0);
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

        if (AutogradTape.Current is { IsExecutingBackward: false } tape)
        {
            if (logits.Grad == null)
                logits.Grad = Tensor<float>.Zeros(logits.Shape);

            ComputeCategoricalCrossEntropyGradient(logits, targets, logits.Grad, labelSmoothing, 1.0f, accumulate: true);
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
        int batch = logits.Shape.Rank > 1 ? logits.Shape[0] : 1;
        int classes = logits.Shape.Rank > 1 ? logits.Shape[1] : logits.Shape[0];

        using var lContig = logits.IsContiguous ? null : logits.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var logSpan = (lContig ?? logits).AsSpan();
        var tgtSpan = (tContig ?? targets).AsSpan();
        var outSpan = gradOutput.AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        double smoothing = labelSmoothing;
        double smoothVal = smoothing / classes;
        double dScale = (double)dLoss / batch;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;
            ReadOnlySpan<float> rowLogits = logSpan.Slice(offset, classes);
            ComputeSplitSumSoftmaxStats(rowLogits, out double maxLogit, out int argMax, out double sumOther);
            double invDenom = 1.0 / (1.0 + sumOther);

            int targetClass = isOneHot ? -1 : (int)tgtSpan[b];
            for (int c = 0; c < classes; c++)
            {
                double y = isOneHot ? tgtSpan[offset + c] : (targetClass == c ? 1.0 : 0.0);
                double smoothedY = (1.0 - smoothing) * y + smoothVal;
                double diff = (c == argMax)
                    ? (((1.0 - smoothedY) - smoothedY * sumOther) * invDenom)
                    : (Math.Exp((double)rowLogits[c] - maxLogit) * invDenom - smoothedY);
                float grad = (float)(dScale * diff);
                if (accumulate)
                    outSpan[offset + c] += grad;
                else
                    outSpan[offset + c] = grad;
            }
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
        int batch = logits.Shape.Rank > 1 ? logits.Shape[0] : 1;
        int classes = logits.Shape.Rank > 1 ? logits.Shape[1] : logits.Shape[0];

        if (batch <= 0 || classes <= 0)
            throw new ArgumentException("Invalid logits tensor dimensions.");

        using var lContig = logits.IsContiguous ? null : logits.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var logSpan = (lContig ?? logits).AsSpan();
        var tgtSpan = (tContig ?? targets).AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        float totalLoss = 0f;

        Span<float> probs = stackalloc float[Math.Min(classes, 256)];
        bool useHeap = classes > 256;
        float[]? heapArr = useHeap ? new float[classes] : null;
        Span<float> pSpan = useHeap ? heapArr.AsSpan() : probs;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;

            float maxLogit = logSpan[offset];
            for (int c = 1; c < classes; c++)
            {
                if (logSpan[offset + c] > maxLogit) maxLogit = logSpan[offset + c];
            }

            float sumExp = 0f;
            for (int c = 0; c < classes; c++)
            {
                pSpan[c] = MathF.Exp(logSpan[offset + c] - maxLogit);
                sumExp += pSpan[c];
            }

            float invSum = 1.0f / sumExp;
            int targetClass = isOneHot ? -1 : (int)tgtSpan[b];
            for (int c = 0; c < classes; c++)
            {
                pSpan[c] *= invSum;
                float y = isOneHot ? tgtSpan[offset + c] : (targetClass == c ? 1.0f : 0.0f);
                float diff = pSpan[c] - y;
                totalLoss += diff * diff;
            }
        }

        totalLoss /= batch;
        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = totalLoss;
        lossTensor.RequiresGrad = logits.RequiresGrad;

        if (AutogradTape.Current is { IsExecutingBackward: false } tape)
        {
            if (logits.Grad == null)
                logits.Grad = Tensor<float>.Zeros(logits.Shape);

            ComputeBrierScoreGradient(logits, targets, logits.Grad, 1.0f, accumulate: true);
            tape.Record(new TapeEntry(AutogradOp.BrierScoreLoss, lossTensor, logits, targets));
        }

        return (totalLoss, lossTensor);
    }

    internal static void ComputeBrierScoreGradient(
        Tensor<float> logits,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss,
        bool accumulate = false)
    {
        int batch = logits.Shape.Rank > 1 ? logits.Shape[0] : 1;
        int classes = logits.Shape.Rank > 1 ? logits.Shape[1] : logits.Shape[0];

        using var lContig = logits.IsContiguous ? null : logits.Contiguous();
        using var tContig = targets.IsContiguous ? null : targets.Contiguous();
        var logSpan = (lContig ?? logits).AsSpan();
        var tgtSpan = (tContig ?? targets).AsSpan();
        var outSpan = gradOutput.AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        float scale = (2.0f * dLoss) / batch;

        Span<float> probs = stackalloc float[Math.Min(classes, 256)];
        bool useHeap = classes > 256;
        float[]? heapArr = useHeap ? new float[classes] : null;
        Span<float> pSpan = useHeap ? heapArr.AsSpan() : probs;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;

            float maxLogit = logSpan[offset];
            for (int c = 1; c < classes; c++)
            {
                if (logSpan[offset + c] > maxLogit) maxLogit = logSpan[offset + c];
            }

            float sumExp = 0f;
            for (int c = 0; c < classes; c++)
            {
                pSpan[c] = MathF.Exp(logSpan[offset + c] - maxLogit);
                sumExp += pSpan[c];
            }

            float invSum = 1.0f / sumExp;
            int targetClass = isOneHot ? -1 : (int)tgtSpan[b];
            float dotDiffProb = 0f;

            for (int c = 0; c < classes; c++)
            {
                pSpan[c] *= invSum;
                float y = isOneHot ? tgtSpan[offset + c] : (targetClass == c ? 1.0f : 0.0f);
                dotDiffProb += (pSpan[c] - y) * pSpan[c];
            }

            for (int c = 0; c < classes; c++)
            {
                float y = isOneHot ? tgtSpan[offset + c] : (targetClass == c ? 1.0f : 0.0f);
                float grad = scale * pSpan[c] * ((pSpan[c] - y) - dotDiffProb);
                if (accumulate)
                    outSpan[offset + c] += grad;
                else
                    outSpan[offset + c] = grad;
            }
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
