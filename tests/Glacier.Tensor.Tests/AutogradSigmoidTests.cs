using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Xunit;

namespace Glacier.Tensor.Tests;

public class AutogradSigmoidTests
{
    [Fact]
    public void Autograd_Sigmoid_PropagatesGradientsAndPreservesOutput()
    {
        using var tape = new AutogradTape();
        using var x = Tensor<float>.FromArray([-2.0f, -0.5f, 0.0f, 1.0f, 2.5f], 5);
        tape.Watch(x);

        // y = sigmoid(x)
        using var y = TensorOps.Sigmoid(x);

        // Verify output is valid before backward
        Assert.Equal(5, y.ElementCount);
        float origY0 = y[0];

        // Seed with ones dy = 1.0 -> dx = dy * y * (1 - y)
        tape.Backward(y, retainGraph: false);

        // CRITICAL CHECK: y must NOT be prematurely disposed by Backward!
        Assert.Equal(origY0, y[0]);
        Assert.False(y.MemoryBlock.IsDisposed);

        // Verify gradients against theoretical formula: s(x) * (1 - s(x))
        Assert.NotNull(x.Grad);
        for (int i = 0; i < x.ElementCount; i++)
        {
            float s = 1.0f / (1.0f + MathF.Exp(-x[i]));
            float expectedGrad = s * (1.0f - s);
            float actualGrad = x.Grad[i];
            Assert.True(MathF.Abs(actualGrad - expectedGrad) < 1e-5f,
                $"Mismatch at {i}: expected {expectedGrad:F5}, got {actualGrad:F5}");
        }
    }
}
