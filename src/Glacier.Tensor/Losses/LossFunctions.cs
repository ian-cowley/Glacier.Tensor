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
}
