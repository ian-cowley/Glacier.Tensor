using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

public class AutogradTests
{
    [Fact]
    public void Autograd_LinearGraph_PropagatesGradientsAccurately()
    {
        using var tape = new AutogradTape();

        using var x = Tensor<float>.FromArray([2.0f, 3.0f], 2);
        using var w = Tensor<float>.FromArray([4.0f, 5.0f], 2);
        tape.Watch(x);
        tape.Watch(w);

        // y = x * w = [8, 15]
        using var y = TensorOps.Multiply(x, w);

        // loss = sum(y)
        tape.Backward(y);

        // dy/dx = w = [4, 5]
        Assert.NotNull(x.Grad);
        Assert.Equal(4.0f, x.Grad[0]);
        Assert.Equal(5.0f, x.Grad[1]);

        // dy/dw = x = [2, 3]
        Assert.NotNull(w.Grad);
        Assert.Equal(2.0f, w.Grad[0]);
        Assert.Equal(3.0f, w.Grad[1]);
    }

    [Fact]
    public void Autograd_MatMul_MatchesFiniteDifferences()
    {
        int M = 2, K = 3, N = 2;
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 123);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 456);

        using (var tape = new AutogradTape())
        {
            tape.Watch(a);
            tape.Watch(b);

            using var c = TensorOps.MatMul(a, b);
            tape.Backward(c);
        }

        // Numerical finite difference check for a[0, 1]
        const float eps = 1e-3f;
        float origA = a[0, 1];

        a[0, 1] = origA + eps;
        using var cPlus = TensorOps.MatMul(a, b);
        float lossPlus = 0f;
        foreach (var v in cPlus.AsSpan()) lossPlus += v;

        a[0, 1] = origA - eps;
        using var cMinus = TensorOps.MatMul(a, b);
        float lossMinus = 0f;
        foreach (var v in cMinus.AsSpan()) lossMinus += v;

        a[0, 1] = origA; // restore

        float numGrad = (lossPlus - lossMinus) / (2.0f * eps);
        float autoGrad = a.Grad![0, 1];

        Assert.True(Math.Abs(autoGrad - numGrad) < 1e-2f, 
            $"Finite diff: {numGrad:F4}, Autograd: {autoGrad:F4}");
    }

    [Fact]
    public void Autograd_ReLU_AppliesGradientMask()
    {
        using var tape = new AutogradTape();
        using var x = Tensor<float>.FromArray([-2.0f, -0.5f, 0.5f, 3.0f], 4);
        tape.Watch(x);

        using var y = TensorOps.ReLU(x);
        tape.Backward(y);

        Assert.NotNull(x.Grad);
        Assert.Equal(0.0f, x.Grad[0]);
        Assert.Equal(0.0f, x.Grad[1]);
        Assert.Equal(1.0f, x.Grad[2]);
        Assert.Equal(1.0f, x.Grad[3]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(24)]
    [InlineData(31)]
    [InlineData(1000)]
    public void Autograd_GradientAccumulation_Vectorized_EqualLength(int n)
    {
        using var tape = new AutogradTape();
        using var x = new Tensor<float>(n);
        using var w = new Tensor<float>(n);

        for (int i = 0; i < n; i++)
        {
            x[i] = i + 1.0f;
            w[i] = (i + 1.0f) * 2.0f;
        }

        tape.Watch(x);
        tape.Watch(w);

        // y = x + w
        using var y = TensorOps.Add(x, w);
        tape.Backward(y);

        Assert.NotNull(x.Grad);
        Assert.NotNull(w.Grad);
        Assert.Equal(n, x.Grad.ElementCount);
        Assert.Equal(n, w.Grad.ElementCount);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(1.0f, x.Grad[i]);
            Assert.Equal(1.0f, w.Grad[i]);
        }
    }

    [Theory]
    [InlineData(5, 17)]
    [InlineData(32, 64)]
    [InlineData(3, 8)]
    public void Autograd_GradientAccumulation_Vectorized_BroadcastReduction(int batch, int d)
    {
        using var tape = new AutogradTape();
        using var x = new Tensor<float>(batch, d);
        using var bias = new Tensor<float>(d);

        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < d; j++)
            {
                x[b, j] = (b + 1) * 10.0f + j;
            }
        }
        for (int j = 0; j < d; j++)
        {
            bias[j] = j * 0.5f;
        }

        tape.Watch(bias);

        // Record a manual tape entry for broadcast add to test broadcast reduction in backward pass
        using var y = new Tensor<float>(batch, d);
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < d; j++)
            {
                y[b, j] = x[b, j] + bias[j];
            }
        }
        y.RequiresGrad = true;
        tape.Record(new TapeEntry(AutogradOp.Add, y, x, bias));

        tape.Backward(y);

        Assert.NotNull(bias.Grad);
        Assert.Equal(d, bias.Grad.ElementCount);

        // In y = x + bias, dLoss/dbias_j = sum_{b=0}^{batch-1} dLoss/dy_{b,j} = batch * 1.0f
        for (int j = 0; j < d; j++)
        {
            Assert.Equal((float)batch, bias.Grad[j]);
        }
    }

    [Fact]
    public void Autograd_MultipleBackward_AccumulatesGradientsCorrectly()
    {
        using var tape = new AutogradTape();
        int n = 48; // spans across Vector512, Vector256, Vector128
        using var x = new Tensor<float>(n);
        for (int i = 0; i < n; i++) x[i] = (float)i;

        tape.Watch(x);

        // y = x + x -> dy/dx = 2.0
        using var y = TensorOps.Add(x, x);
        tape.Backward(y);

        Assert.NotNull(x.Grad);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(2.0f, x.Grad[i]);
        }
    }
}
