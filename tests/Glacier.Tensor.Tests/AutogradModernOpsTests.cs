using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
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
        tape.Backward(y, retainGraph: false);

        Assert.NotNull(x.Grad);
        Assert.NotNull(w.Grad);

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

    [Fact]
    public void Autograd_Softmax_InnerAndOuterDims_MatchFiniteDifferences()
    {
        const float epsDiff = 1e-3f;
        float[] xInit = [0.7f, -1.1f, 0.3f, 1.4f, -0.4f, 0.9f];
        float[] weights = [0.2f, -0.5f, 1.1f, -0.3f, 0.8f, 0.4f];

        foreach (int dim in new[] { 0, 1 })
        {
            using var x = Tensor<float>.FromArray(xInit, 2, 3);
            using var w = Tensor<float>.FromArray(weights, 2, 3);
            using var tape = new AutogradTape();
            tape.Watch(x);

            using var probs = TensorOps.Softmax(x, dim);
            using var prod = TensorOps.Multiply(probs, w);
            tape.Backward(prod);

            Assert.NotNull(x.Grad);

            for (int idx = 0; idx < xInit.Length; idx++)
            {
                float orig = xInit[idx];
                xInit[idx] = orig + epsDiff;
                using var xp = Tensor<float>.FromArray(xInit, 2, 3);
                using var yp = TensorOps.Softmax(xp, dim);
                float lp = 0f;
                for (int k = 0; k < weights.Length; k++) lp += yp.AsSpan()[k] * weights[k];

                xInit[idx] = orig - epsDiff;
                using var xm = Tensor<float>.FromArray(xInit, 2, 3);
                using var ym = TensorOps.Softmax(xm, dim);
                float lm = 0f;
                for (int k = 0; k < weights.Length; k++) lm += ym.AsSpan()[k] * weights[k];

                xInit[idx] = orig;
                float numGrad = (lp - lm) / (2.0f * epsDiff);
                float anaGrad = x.Grad.AsSpan()[idx];
                Assert.True(MathF.Abs(anaGrad - numGrad) < 2e-3f,
                    $"Softmax dim={dim} mismatch at {idx}: analytical={anaGrad:F6}, numerical={numGrad:F6}");
            }
        }
    }

    [Fact]
    public void Autograd_LayerNorm_MatchesFiniteDifferences()
    {
        const int batch = 2;
        const int dim = 4;
        const float epsDiff = 1e-3f;

        float[] xInit = [1.2f, -0.5f, 2.1f, 0.8f, -1.5f, 0.3f, -0.7f, 1.9f];
        float[] outWeights = [0.4f, -0.9f, 1.3f, 0.2f, -0.6f, 1.1f, 0.5f, -0.3f];

        using var ln = new LayerNorm(dim, 1e-5f);
        ln.Gamma.AsSpan()[0] = 1.3f;
        ln.Gamma.AsSpan()[1] = 0.7f;
        ln.Beta.AsSpan()[0] = -0.2f;
        ln.Beta.AsSpan()[2] = 0.4f;

        using var x = Tensor<float>.FromArray(xInit, batch, dim);
        using var wOut = Tensor<float>.FromArray(outWeights, batch, dim);

        using var tape = new AutogradTape();
        tape.Watch(x);
        ln.ZeroGrad();

        using var y = ln.Forward(x);
        using var weighted = TensorOps.Multiply(y, wOut);
        tape.Backward(weighted);

        Assert.NotNull(x.Grad);
        Assert.NotNull(ln.Gamma.Grad);
        Assert.NotNull(ln.Beta.Grad);

        for (int idx = 0; idx < xInit.Length; idx++)
        {
            float orig = xInit[idx];
            xInit[idx] = orig + epsDiff;
            using var xp = Tensor<float>.FromArray(xInit, batch, dim);
            using var yp = ln.Forward(xp);
            float lp = 0f;
            for (int k = 0; k < outWeights.Length; k++) lp += yp.AsSpan()[k] * outWeights[k];

            xInit[idx] = orig - epsDiff;
            using var xm = Tensor<float>.FromArray(xInit, batch, dim);
            using var ym = ln.Forward(xm);
            float lm = 0f;
            for (int k = 0; k < outWeights.Length; k++) lm += ym.AsSpan()[k] * outWeights[k];

            xInit[idx] = orig;
            float numGrad = (lp - lm) / (2.0f * epsDiff);
            float anaGrad = x.Grad.AsSpan()[idx];
            Assert.True(MathF.Abs(anaGrad - numGrad) < 3e-3f,
                $"LayerNorm dx mismatch at {idx}: analytical={anaGrad:F6}, numerical={numGrad:F6}");
        }
    }

    [Fact]
    public void Autograd_ViewOpsAndNestedTape_PropagateAccurately()
    {
        float[] init = [1f, 2f, 3f, 4f, 5f, 6f];
        using var x = Tensor<float>.FromArray(init, 2, 3);

        using var outerTape = new AutogradTape();
        outerTape.Watch(x);

        using (var innerTape = new AutogradTape())
        {
            Assert.Same(innerTape, AutogradTape.Current);
            using var tmp = Tensor<float>.FromArray([2f, 3f], 2);
            innerTape.Watch(tmp);
            using var innerScaled = TensorOps.Scale(tmp, 4.0f);
            innerTape.Backward(innerScaled);
            Assert.Equal(4.0f, tmp.Grad![0]);
        }

        Assert.Same(outerTape, AutogradTape.Current);

        using var sliced = x.Slice(1, 1, 2); // [2, 2] -> [[2,3],[5,6]]
        using var transposed = sliced.Transpose(); // [2, 2] -> [[2,5],[3,6]]
        using var reshaped = transposed.Reshape(4);
        using var scaled = TensorOps.Scale(reshaped, 3.0f);
        outerTape.Backward(scaled);

        Assert.NotNull(x.Grad);
        // Column 0 was excluded from slice -> grad 0; columns 1 & 2 scaled by 3 -> grad 3
        Assert.Equal(0f, x.Grad[0, 0]);
        Assert.Equal(3f, x.Grad[0, 1]);
        Assert.Equal(3f, x.Grad[0, 2]);
        Assert.Equal(0f, x.Grad[1, 0]);
        Assert.Equal(3f, x.Grad[1, 1]);
        Assert.Equal(3f, x.Grad[1, 2]);
    }

    [Fact]
    public void Autograd_LossFunctions_BackwardThroughLossTensor_MatchesAnalyticalGradients()
    {
        float[] wInit = [0.3f, -0.4f, 0.2f, 0.5f, -0.1f, 0.7f];
        float[] xInit = [1.0f, -0.5f, 0.8f, -1.2f];
        float[] targetsInit = [1f, 0f, 0f, 0f, 1f, 0f];

        using var x = Tensor<float>.FromArray(xInit, 2, 2);
        using var targets = Tensor<float>.FromArray(targetsInit, 2, 3);
        using var w = Tensor<float>.FromArray(wInit, 2, 3);

        using var tape = new AutogradTape();
        tape.Watch(w);

        using var logits = TensorOps.MatMul(x, w);
        var (_, lossTensor) = LossFunctions.BrierScoreLoss(logits, targets);
        using (lossTensor)
        {
            tape.Backward(lossTensor);
        }

        Assert.NotNull(w.Grad);
        const float eps = 1e-3f;
        for (int idx = 0; idx < wInit.Length; idx++)
        {
            float orig = wInit[idx];
            wInit[idx] = orig + eps;
            using var wp = Tensor<float>.FromArray(wInit, 2, 3);
            using var lpTensor = TensorOps.MatMul(x, wp);
            var (lp, ltP) = LossFunctions.BrierScoreLoss(lpTensor, targets);
            ltP.Dispose();

            wInit[idx] = orig - eps;
            using var wm = Tensor<float>.FromArray(wInit, 2, 3);
            using var lmTensor = TensorOps.MatMul(x, wm);
            var (lm, ltM) = LossFunctions.BrierScoreLoss(lmTensor, targets);
            ltM.Dispose();

            wInit[idx] = orig;
            float numGrad = (lp - lm) / (2.0f * eps);
            float anaGrad = w.Grad.AsSpan()[idx];
            Assert.True(MathF.Abs(anaGrad - numGrad) < 2e-3f,
                $"BrierScore dW mismatch at {idx}: analytical={anaGrad:F6}, numerical={numGrad:F6}");
        }
    }
}
