using System;
using System.Reflection;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests.StressChallenge;

public class AutogradVectorizationStressTests
{
    private static readonly MethodInfo s_accumulateGradient = typeof(AutogradTape)
        .GetMethod("AccumulateGradient", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("Method AccumulateGradient not found on AutogradTape.");

    private static void AccumulateGradientDirect(Tensor<float> target, Tensor<float> delta)
    {
        s_accumulateGradient.Invoke(null, [target, delta]);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(1024)]
    [InlineData(65536)]
    public void Autograd_VectorizedAccumulation_MatchesMathematicalOracle(int length)
    {
        // 1. Arrange tensors with known non-zero pseudo-random values
        var rng = new Random(1000 + length);
        using var target = new Tensor<float>(length);
        using var delta = new Tensor<float>(length);

        target.RequiresGrad = true;
        target.Grad = new Tensor<float>(length);

        float[] initialTarget = new float[length];
        float[] deltaValues = new float[length];
        float[] oracleResult = new float[length];

        var tSpan = target.Grad.AsSpan();
        var dSpan = delta.AsSpan();

        for (int i = 0; i < length; i++)
        {
            // Use floating point numbers that exercise exponent and mantissa bits
            float tVal = (float)(rng.NextDouble() * 200.0 - 100.0);
            float dVal = (float)(rng.NextDouble() * 200.0 - 100.0);

            initialTarget[i] = tVal;
            deltaValues[i] = dVal;
            tSpan[i] = tVal;
            dSpan[i] = dVal;

            // Mathematical oracle: pure scalar addition in double precision, cast to float
            oracleResult[i] = (float)((double)tVal + (double)dVal);
        }

        // 2. Act: invoke vectorized gradient accumulation kernel
        AccumulateGradientDirect(target, delta);

        // 3. Assert: verify every single float matches oracle exactly or within 1 ULP
        var actualSpan = target.Grad.AsSpan();
        for (int i = 0; i < length; i++)
        {
            float actual = actualSpan[i];
            float expected = oracleResult[i];
            float diff = Math.Abs(actual - expected);

            Assert.True(diff <= 1e-5f,
                $"Discrepancy at index {i}/{length}: Expected {expected:F6}, Actual {actual:F6}, Diff {diff:E4}");
        }
    }

    [Theory]
    [InlineData(7, 10)]
    [InlineData(15, 10)]
    [InlineData(31, 10)]
    [InlineData(63, 10)]
    [InlineData(1024, 10)]
    public void Autograd_MultiPassAccumulation_MatchesAccumulatedOracle(int length, int passes)
    {
        var rng = new Random(2000 + length);
        using var target = new Tensor<float>(length);
        target.RequiresGrad = true;
        target.Grad = new Tensor<float>(length);

        double[] oracle = new double[length];
        var targetSpan = target.Grad.AsSpan();

        // Initialize target
        for (int i = 0; i < length; i++)
        {
            float v = (float)(rng.NextDouble() * 50.0 - 25.0);
            targetSpan[i] = v;
            oracle[i] = v;
        }

        // Multi-pass gradient accumulation
        for (int p = 0; p < passes; p++)
        {
            using var delta = new Tensor<float>(length);
            var deltaSpan = delta.AsSpan();
            for (int i = 0; i < length; i++)
            {
                float d = (float)(rng.NextDouble() * 20.0 - 10.0);
                deltaSpan[i] = d;
                oracle[i] += d;
            }

            AccumulateGradientDirect(target, delta);
        }

        // Verify multi-pass accumulation fidelity
        var finalSpan = target.Grad.AsSpan();
        for (int i = 0; i < length; i++)
        {
            float actual = finalSpan[i];
            float expected = (float)oracle[i];
            float diff = Math.Abs(actual - expected);

            Assert.True(diff <= 1e-4f,
                $"Multi-pass discrepancy at index {i}/{length}: Expected {expected:F6}, Actual {actual:F6}, Diff {diff:E4}");
        }
    }

    [Theory]
    [InlineData(1, 1000)]
    [InlineData(7, 5)]
    [InlineData(7, 13)]
    [InlineData(8, 16)]
    [InlineData(15, 4)]
    [InlineData(16, 32)]
    [InlineData(31, 3)]
    [InlineData(32, 64)]
    [InlineData(63, 7)]
    [InlineData(64, 16)]
    [InlineData(1024, 64)]
    public void Autograd_BroadcastReduction_MatchesMathematicalOracle(int d, int batch)
    {
        int deltaLength = d * batch;
        var rng = new Random(3000 + d + batch);

        using var target = new Tensor<float>(d);
        using var delta = new Tensor<float>(batch, d);

        target.RequiresGrad = true;
        target.Grad = new Tensor<float>(d);

        double[] oracle = new double[d];
        var targetSpan = target.Grad.AsSpan();
        for (int i = 0; i < d; i++)
        {
            float v = (float)(rng.NextDouble() * 10.0 - 5.0);
            targetSpan[i] = v;
            oracle[i] = v;
        }

        var deltaSpan = delta.AsSpan();
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < d; i++)
            {
                float dVal = (float)(rng.NextDouble() * 10.0 - 5.0);
                deltaSpan[b * d + i] = dVal;
                oracle[i] += dVal;
            }
        }

        // Execute broadcast reduction accumulation
        AccumulateGradientDirect(target, delta);

        // Verify against broadcast reduction mathematical oracle
        var resultSpan = target.Grad.AsSpan();
        for (int i = 0; i < d; i++)
        {
            float actual = resultSpan[i];
            float expected = (float)oracle[i];
            float diff = Math.Abs(actual - expected);

            // Numerical tolerance scaled by sqrt(batch) for floating point addition order
            float tolerance = Math.Max(1e-4f, 1e-4f * (float)Math.Sqrt(batch));
            Assert.True(diff <= tolerance,
                $"Broadcast reduction mismatch at index {i}/{d} (batch={batch}): Expected {expected:F6}, Actual {actual:F6}, Diff {diff:E4}");
        }
    }

    [Fact]
    public void Autograd_FullTapeBackward_MultipleBranches_AccumulatesVectorizedGradients()
    {
        const int length = 1024;
        using var tape = new AutogradTape();

        using var x = new Tensor<float>(length);
        using var w1 = new Tensor<float>(length);
        using var w2 = new Tensor<float>(length);

        x.RequiresGrad = true;
        w1.RequiresGrad = false;
        w2.RequiresGrad = false;

        tape.Watch(x);

        var xSpan = x.AsSpan();
        var w1Span = w1.AsSpan();
        var w2Span = w2.AsSpan();

        for (int i = 0; i < length; i++)
        {
            xSpan[i] = 2.0f;
            w1Span[i] = (float)(i + 1);
            w2Span[i] = (float)((i + 1) * 2);
        }

        // Branch 1: y1 = x * w1
        using var y1 = TensorOps.Multiply(x, w1);
        // Branch 2: y2 = x * w2
        using var y2 = TensorOps.Multiply(x, w2);
        // Combined: z = y1 + y2
        using var z = TensorOps.Add(y1, y2);

        // Reverse-mode automatic differentiation
        tape.Backward(z);

        // Expected gradient: dz/dx = w1 + w2 = 3 * (i + 1)
        Assert.NotNull(x.Grad);
        var gradSpan = x.Grad.AsSpan();
        for (int i = 0; i < length; i++)
        {
            float expected = (float)(i + 1) * 3.0f;
            float actual = gradSpan[i];
            Assert.Equal(expected, actual);
        }
    }
}
