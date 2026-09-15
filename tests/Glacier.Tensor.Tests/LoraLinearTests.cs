using System;
using System.Linq;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;
using Xunit;

namespace Glacier.Tensor.Tests;

public class LoraLinearTests
{
    [Fact]
    public void LoraLinear_InitialOutput_MatchesBaseModelExactly()
    {
        int inFeatures = 16;
        int outFeatures = 8;
        int batch = 4;

        using var baseW = TensorFloatExtensions.RandomUniform([inFeatures, outFeatures], -1f, 1f, seed: 42);
        using var lora = new LoraLinear(baseW, null, rank: 4, alpha: 8f, seed: 100);
        using var x = TensorFloatExtensions.RandomUniform([batch, inFeatures], -1f, 1f, seed: 200);

        // Expected: X * W0 (since B is initialized to 0)
        using var expected = TensorOps.MatMul(x, baseW);
        using var actual = lora.Forward(x);

        var expSpan = expected.AsSpan();
        var actSpan = actual.AsSpan();

        Assert.Equal(expSpan.Length, actSpan.Length);
        for (int i = 0; i < expSpan.Length; i++)
        {
            Assert.Equal(expSpan[i], actSpan[i], 5);
        }
    }

    [Fact]
    public void LoraLinear_Gradients_FlowOnlyToAdapters_BaseRemainsFrozen()
    {
        int inFeatures = 8;
        int outFeatures = 4;
        int batch = 2;

        using var baseW = TensorFloatExtensions.RandomUniform([inFeatures, outFeatures], -1f, 1f, seed: 42);
        using var lora = new LoraLinear(baseW, null, rank: 2, alpha: 4f, seed: 100);
        using var x = TensorFloatExtensions.RandomUniform([batch, inFeatures], -1f, 1f, seed: 200);
        using var targets = TensorFloatExtensions.Ones(batch, outFeatures);

        using var tape = new AutogradTape();
        foreach (var p in lora.Parameters) tape.Watch(p);

        using var output = lora.Forward(x);
        var (loss, _) = LossFunctions.MSELoss(output, targets);

        tape.Backward(output);

        // Base weight must have no gradients and remain RequiresGrad = false
        Assert.False(lora.BaseWeight.RequiresGrad);
        Assert.True(lora.BaseWeight.Grad == null || lora.BaseWeight.Grad.AsSpan().ToArray().All(g => g == 0f));

        // Adapters must receive valid gradients
        Assert.NotNull(lora.AdapterA.Grad);
        Assert.NotNull(lora.AdapterB.Grad);

        // Adapter B received non-zero gradients
        bool bHasGradients = lora.AdapterB.Grad.AsSpan().ToArray().Any(g => Math.Abs(g) > 1e-6f);
        Assert.True(bHasGradients, "Adapter B should receive non-zero gradients");
    }

    [Fact]
    public void LoraLinear_AdamW_FineTuning_ReducesLossAndAdapts()
    {
        int inFeatures = 16;
        int outFeatures = 8;
        int batch = 4;

        using var baseW = TensorFloatExtensions.RandomUniform([inFeatures, outFeatures], -0.5f, 0.5f, seed: 42);
        // Make a clone of baseW to verify base weights NEVER change
        var baseWOriginal = baseW.AsSpan().ToArray();

        using var lora = new LoraLinear(baseW, null, rank: 4, alpha: 8f, seed: 77);
        using var optimizer = new AdamW(lora.Parameters, lr: 0.05f);

        using var x = TensorFloatExtensions.RandomUniform([batch, inFeatures], -1f, 1f, seed: 88);
        using var target = TensorFloatExtensions.RandomUniform([batch, outFeatures], -2f, 2f, seed: 99);

        float initialLoss = 0f;
        float finalLoss = 0f;

        for (int step = 0; step < 30; step++)
        {
            optimizer.ZeroGrad();

            using var tape = new AutogradTape();
            foreach (var p in lora.Parameters) tape.Watch(p);

            using var pred = lora.Forward(x);
            var (lossVal, _) = LossFunctions.MSELoss(pred, target);

            if (step == 0) initialLoss = lossVal;
            finalLoss = lossVal;

            tape.Backward(pred);
            optimizer.Step();
        }

        // 1. Loss must drop significantly (> 50%)
        Assert.True(finalLoss < initialLoss * 0.5f,
            $"Fine-tuning did not converge: Initial={initialLoss:F4}, Final={finalLoss:F4}");

        // 2. Base weight must remain completely bit-for-bit identical to original
        var baseWCurrent = baseW.AsSpan();
        for (int i = 0; i < baseWOriginal.Length; i++)
        {
            Assert.Equal(baseWOriginal[i], baseWCurrent[i]);
        }
    }

    [Fact]
    public void LoraLinear_Merge_ProducesEquivalentOutputToAdapterForward()
    {
        int inFeatures = 16;
        int outFeatures = 8;
        int batch = 4;

        using var baseW = TensorFloatExtensions.RandomUniform([inFeatures, outFeatures], -0.5f, 0.5f, seed: 12);
        using var lora = new LoraLinear(baseW, null, rank: 4, alpha: 8f, seed: 34);

        // Perturb adapter B so delta is non-zero
        var bSpan = lora.AdapterB.AsSpan();
        for (int i = 0; i < bSpan.Length; i++)
        {
            bSpan[i] = 0.1f * (i + 1);
        }

        using var x = TensorFloatExtensions.RandomUniform([batch, inFeatures], -1f, 1f, seed: 56);

        // 1. Forward with LoRA adapters
        using var loraOutput = lora.Forward(x);

        // 2. Merge adapters into base weight
        using var mergedW = lora.Merge();

        // 3. Forward with single merged weight (zero-overhead inference)
        using var mergedOutput = TensorOps.MatMul(x, mergedW);

        var loraSpan = loraOutput.AsSpan();
        var mergedSpan = mergedOutput.AsSpan();

        Assert.Equal(loraSpan.Length, mergedSpan.Length);
        for (int i = 0; i < loraSpan.Length; i++)
        {
            Assert.Equal(loraSpan[i], mergedSpan[i], 4);
        }
    }
}
