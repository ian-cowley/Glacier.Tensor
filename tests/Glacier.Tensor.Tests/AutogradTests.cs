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
}
