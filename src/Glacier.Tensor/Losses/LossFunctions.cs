using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Losses;

/// <summary>
/// Differentiable loss functions for deep learning training.
/// </summary>
public static class LossFunctions
{
    public static (float lossValue, Tensor<float> lossTensor) MSELoss(Tensor<float> predictions, Tensor<float> targets)
    {
        if (predictions.ElementCount != targets.ElementCount)
            throw new ArgumentException("Prediction and target shapes must match.");

        int n = (int)predictions.ElementCount;
        var pSpan = predictions.AsSpan();
        var tSpan = targets.AsSpan();
        float totalLoss = 0f;
        for (int i = 0; i < n; i++)
        {
            float d = pSpan[i] - tSpan[i];
            totalLoss += d * d;
        }
        totalLoss /= n;

        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = totalLoss;

        if (AutogradTape.Current != null)
        {
            if (predictions.Grad == null)
                predictions.Grad = Tensor<float>.Zeros(predictions.Shape);

            float scale = 2.0f / n;
            var gSpan = predictions.Grad.AsSpan();
            for (int i = 0; i < n; i++)
            {
                gSpan[i] += scale * (pSpan[i] - tSpan[i]);
            }
        }

        return (totalLoss, lossTensor);
    }

    public static (float lossValue, Tensor<float> lossTensor) BinaryCrossEntropy(Tensor<float> predictions, Tensor<float> targets)
    {
        if (predictions.ElementCount != targets.ElementCount)
            throw new ArgumentException("Prediction and target shapes must match.");

        int n = (int)predictions.ElementCount;
        var pSpan = predictions.AsSpan();
        var tSpan = targets.AsSpan();
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

        if (AutogradTape.Current != null)
        {
            if (predictions.Grad == null)
                predictions.Grad = Tensor<float>.Zeros(predictions.Shape);

            var gSpan = predictions.Grad.AsSpan();
            float invN = 1.0f / n;
            for (int i = 0; i < n; i++)
            {
                float p = Math.Clamp(pSpan[i], eps, 1.0f - eps);
                float y = tSpan[i];
                gSpan[i] += invN * (p - y) / (p * (1.0f - p));
            }
        }

        return (totalLoss, lossTensor);
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

        var logSpan = logits.AsSpan();
        var tgtSpan = targets.AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;

        float totalLoss = 0f;
        const float eps = 1e-12f;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;

            // 1. Find max logit for numerical stability
            float maxLogit = logSpan[offset];
            for (int c = 1; c < classes; c++)
            {
                if (logSpan[offset + c] > maxLogit)
                    maxLogit = logSpan[offset + c];
            }

            // 2. Sum exp(z - max)
            float sumExp = 0f;
            for (int c = 0; c < classes; c++)
            {
                sumExp += MathF.Exp(logSpan[offset + c] - maxLogit);
            }

            // 3. Compute loss
            if (isOneHot)
            {
                for (int c = 0; c < classes; c++)
                {
                    float y = tgtSpan[offset + c];
                    if (y > 0f)
                    {
                        float p = MathF.Exp(logSpan[offset + c] - maxLogit) / sumExp;
                        totalLoss -= y * MathF.Log(MathF.Max(p, eps));
                    }
                }
            }
            else
            {
                int targetClass = (int)tgtSpan[b];
                if (targetClass >= 0 && targetClass < classes)
                {
                    float p = MathF.Exp(logSpan[offset + targetClass] - maxLogit) / sumExp;
                    totalLoss -= MathF.Log(MathF.Max(p, eps));
                }
            }
        }

        totalLoss /= batch;

        var lossTensor = new Tensor<float>(1);
        lossTensor.AsSpan()[0] = totalLoss;

        if (AutogradTape.Current != null)
        {
            if (logits.Grad == null)
                logits.Grad = Tensor<float>.Zeros(logits.Shape);

            ComputeCrossEntropyGradient(logits, targets, logits.Grad, 1.0f);

            AutogradTape.Current.Record(new TapeEntry(AutogradOp.CrossEntropyLoss, lossTensor, logits, targets));
        }

        return (totalLoss, lossTensor);
    }

    internal static void ComputeCrossEntropyGradient(
        Tensor<float> logits,
        Tensor<float> targets,
        Tensor<float> gradOutput,
        float dLoss)
    {
        int batch = logits.Shape.Rank > 1 ? logits.Shape[0] : 1;
        int classes = logits.Shape.Rank > 1 ? logits.Shape[1] : logits.Shape[0];

        var logSpan = logits.AsSpan();
        var tgtSpan = targets.AsSpan();
        var outSpan = gradOutput.AsSpan();
        bool isOneHot = targets.ElementCount == (long)batch * classes;
        float scale = dLoss / batch;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * classes;

            // 1. Max logit
            float maxLogit = logSpan[offset];
            for (int c = 1; c < classes; c++)
            {
                if (logSpan[offset + c] > maxLogit)
                    maxLogit = logSpan[offset + c];
            }

            // 2. Sum exp
            float sumExp = 0f;
            for (int c = 0; c < classes; c++)
            {
                sumExp += MathF.Exp(logSpan[offset + c] - maxLogit);
            }

            // 3. Softmax & Gradient: dZ = scale * (p - y)
            int targetClass = isOneHot ? -1 : (int)tgtSpan[b];

            for (int c = 0; c < classes; c++)
            {
                float p = MathF.Exp(logSpan[offset + c] - maxLogit) / sumExp;
                float y = isOneHot ? tgtSpan[offset + c] : (targetClass == c ? 1.0f : 0.0f);
                outSpan[offset + c] = scale * (p - y);
            }
        }
    }
}
