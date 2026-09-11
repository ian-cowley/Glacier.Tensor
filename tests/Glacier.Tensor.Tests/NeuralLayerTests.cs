using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;
using Xunit;

namespace Glacier.Tensor.Tests;

public class NeuralLayerTests
{
    [Fact]
    public void Linear_ForwardPass_ComputesExpectedDimensions()
    {
        using var layer = new Linear(4, 3, seed: 10);
        using var input = TensorFloatExtensions.Ones(2, 4);

        using var output = layer.Forward(input);

        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(3, output.Shape[1]);
    }

    [Fact]
    public void LayerNorm_NormalizesMeanAndVariance()
    {
        using var ln = new LayerNorm(4);
        using var input = Tensor<float>.FromArray([10f, 20f, 30f, 40f], 1, 4);

        using var output = ln.Forward(input);

        var span = output.AsSpan();
        float mean = 0f;
        for (int i = 0; i < 4; i++) mean += span[i];
        mean /= 4f;

        Assert.True(Math.Abs(mean) < 1e-4f, $"LayerNorm mean should be close to 0, was {mean}");
    }

    [Fact]
    public void MLP_OptimizationStep_ReducesLossWithAdamW()
    {
        // 2-layer MLP: [4] -> [8] -> [1]
        using var l1 = new Linear(4, 8, seed: 42);
        using var l2 = new Linear(8, 1, seed: 84);

        var paramsList = new[] { l1.Weight, l1.Bias, l2.Weight, l2.Bias };
        using var optimizer = new AdamW(paramsList, lr: 0.05f);

        using var x = TensorFloatExtensions.RandomUniform([8, 4], -1f, 1f, seed: 100);
        using var targets = TensorFloatExtensions.Ones(8, 1);

        float initialLoss = 0f;
        float finalLoss = 0f;

        for (int epoch = 0; epoch < 25; epoch++)
        {
            optimizer.ZeroGrad();

            using var tape = new AutogradTape();
            foreach (var p in paramsList) tape.Watch(p);

            using var h1 = l1.Forward(x);
            using var a1 = TensorOps.ReLU(h1);
            using var outPred = l2.Forward(a1);

            var (lossVal, lossTensor) = LossFunctions.MSELoss(outPred, targets);

            if (epoch == 0) initialLoss = lossVal;
            finalLoss = lossVal;

            tape.Backward(outPred);
            optimizer.Step();
        }

        Assert.True(finalLoss < initialLoss, 
            $"Training did not reduce loss: Initial={initialLoss:F4}, Final={finalLoss:F4}");
    }
}
