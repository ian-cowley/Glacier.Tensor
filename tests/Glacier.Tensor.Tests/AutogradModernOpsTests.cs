using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

public class AutogradModernOpsTests
{
    [Fact]
    public void Autograd_GELU_MatchesFiniteDifferences()
    {
        float[] values = [-2.5f, -1.0f, -0.2f, 0.0f, 0.5f, 1.2f, 3.0f];
        const float eps = 1e-3f;

        using var x = Tensor<float>.FromArray(values, values.Length);
        using var tape = new AutogradTape();
        tape.Watch(x);

        using var y = TensorOps.GELU(x);
        tape.Backward(y, retainGraph: false);

        Assert.NotNull(x.Grad);

        // Verify each element against central finite difference
        for (int i = 0; i < values.Length; i++)
        {
            float xVal = values[i];

            float xPlus = xVal + eps;
            float uPlus = 0.7978845608f * (xPlus + 0.044715f * xPlus * xPlus * xPlus);
            float geluPlus = 0.5f * xPlus * (1.0f + MathF.Tanh(uPlus));

            float xMinus = xVal - eps;
            float uMinus = 0.7978845608f * (xMinus + 0.044715f * xMinus * xMinus * xMinus);
            float geluMinus = 0.5f * xMinus * (1.0f + MathF.Tanh(uMinus));

            float numericalGrad = (geluPlus - geluMinus) / (2.0f * eps);
            float analyticalGrad = x.Grad[i];

            Assert.True(MathF.Abs(analyticalGrad - numericalGrad) < 1e-3f,
                $"GELU mismatch at {i} (x={xVal}): analytical={analyticalGrad:F6}, numerical={numericalGrad:F6}");
        }
    }

    [Fact]
    public void Autograd_RMSNorm_MatchesFiniteDifferences()
    {
        const int batch = 2;
        const int dim = 4;
        const float epsDiff = 1e-3f;
        const float rmsEps = 1e-5f;

        float[] xInit = [1.2f, -0.5f, 2.1f, 0.8f, -1.5f, 0.3f, -0.7f, 1.9f];
        float[] wInit = [1.0f, 0.8f, 1.2f, 0.5f];

        using var x = Tensor<float>.FromArray(xInit, batch, dim);
        using var w = Tensor<float>.FromArray(wInit, dim);

        using var tape = new AutogradTape();
        tape.Watch(x);
        tape.Watch(w);

        using var y = TensorOps.RMSNorm(x, w, rmsEps);

        // Loss = sum(y), so dy = 1.0 everywhere
        tape.Backward(y, retainGraph: false);

        Assert.NotNull(x.Grad);
        Assert.NotNull(w.Grad);

        // Helper to compute forward loss L = sum(RMSNorm(x, w))
        static float ComputeLoss(float[] xArr, float[] wArr, int bCount, int dCount, float epsVal)
        {
            float total = 0f;
            for (int b = 0; b < bCount; b++)
            {
                int off = b * dCount;
                float sumSq = 0f;
                for (int i = 0; i < dCount; i++)
                {
                    float v = xArr[off + i];
                    sumSq += v * v;
                }
                float rms = MathF.Sqrt(sumSq / dCount + epsVal);
                for (int i = 0; i < dCount; i++)
                {
                    total += (xArr[off + i] / rms) * wArr[i];
                }
            }
            return total;
        }

        // Verify gradient w.r.t x
        for (int idx = 0; idx < xInit.Length; idx++)
        {
            float orig = xInit[idx];

            xInit[idx] = orig + epsDiff;
            float lossPlus = ComputeLoss(xInit, wInit, batch, dim, rmsEps);

            xInit[idx] = orig - epsDiff;
            float lossMinus = ComputeLoss(xInit, wInit, batch, dim, rmsEps);

            xInit[idx] = orig;

            float numGrad = (lossPlus - lossMinus) / (2.0f * epsDiff);
            float anaGrad = x.Grad.AsSpan()[idx];

            Assert.True(MathF.Abs(anaGrad - numGrad) < 2e-3f,
                $"RMSNorm dx mismatch at {idx}: analytical={anaGrad:F6}, numerical={numGrad:F6}");
        }

        // Verify gradient w.r.t w
        for (int idx = 0; idx < wInit.Length; idx++)
        {
            float orig = wInit[idx];

            wInit[idx] = orig + epsDiff;
            float lossPlus = ComputeLoss(xInit, wInit, batch, dim, rmsEps);

            wInit[idx] = orig - epsDiff;
            float lossMinus = ComputeLoss(xInit, wInit, batch, dim, rmsEps);

            wInit[idx] = orig;

            float numGrad = (lossPlus - lossMinus) / (2.0f * epsDiff);
            float anaGrad = w.Grad.AsSpan()[idx];

            Assert.True(MathF.Abs(anaGrad - numGrad) < 2e-3f,
                $"RMSNorm dw mismatch at {idx}: analytical={anaGrad:F6}, numerical={numGrad:F6}");
        }
    }
}
