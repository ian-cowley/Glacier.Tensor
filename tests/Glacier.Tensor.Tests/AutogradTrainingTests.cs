using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;
using Xunit;

namespace Glacier.Tensor.Tests;

public class AutogradTrainingTests
{
    [Fact]
    public void CrossEntropy_AnalyticalGradient_MatchesFiniteDifferences()
    {
        const int B = 3;
        const int C = 4;
        using var logits = TensorFloatExtensions.RandomUniform([B, C], -1.0f, 1.0f, seed: 42);
        using var targets = new Tensor<float>(B, C);

        // One-hot targets: [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]
        targets[0, 0] = 1.0f;
        targets[1, 2] = 1.0f;
        targets[2, 1] = 1.0f;

        // Backward with AutogradTape
        using (var tape = new AutogradTape())
        {
            tape.Watch(logits);
            var (lossVal, lossTensor) = LossFunctions.CrossEntropy(logits, targets);
            tape.Backward(lossTensor);
        }

        Assert.NotNull(logits.Grad);

        // Finite differences verification for each element
        const float eps = 1e-3f;
        for (int b = 0; b < B; b++)
        {
            for (int c = 0; c < C; c++)
            {
                float orig = logits[b, c];

                logits[b, c] = orig + eps;
                var (lossPlus, _) = LossFunctions.CrossEntropy(logits, targets);

                logits[b, c] = orig - eps;
                var (lossMinus, _) = LossFunctions.CrossEntropy(logits, targets);

                logits[b, c] = orig;

                float numGrad = (lossPlus - lossMinus) / (2.0f * eps);
                float analyticalGrad = logits.Grad[b, c];

                Assert.True(MathF.Abs(numGrad - analyticalGrad) < 1e-3f,
                    $"Gradient mismatch at [{b}, {c}]: numerical={numGrad:F5}, analytical={analyticalGrad:F5}");
            }
        }
    }

    [Fact]
    public void CrossEntropy_ClassIndexTargets_ComputesCorrectLossAndGradient()
    {
        const int B = 2;
        const int C = 3;
        using var logits = Tensor<float>.FromArray([
            2.0f, 1.0f, 0.1f,
            0.5f, 2.5f, 0.3f
        ], B, C);

        using var targets = Tensor<float>.FromArray([0f, 1f], B); // class 0 for row 0, class 1 for row 1

        using var tape = new AutogradTape();
        tape.Watch(logits);

        var (lossVal, lossTensor) = LossFunctions.CrossEntropy(logits, targets);
        tape.Backward(lossTensor);

        Assert.True(lossVal > 0f && lossVal < 1.0f, $"Loss should be small since logits favor targets, was {lossVal}");
        Assert.NotNull(logits.Grad);

        // For correct class, gradient = (p - 1) / B, which must be negative
        Assert.True(logits.Grad[0, 0] < 0f, "Correct class 0 logit gradient should be negative");
        Assert.True(logits.Grad[1, 1] < 0f, "Correct class 1 logit gradient should be negative");
    }

    [Fact]
    public void EndToEndTraining_MultiClassClassification_LossConverges()
    {
        // 2-layer Neural Network: [Features=8] -> Linear -> GELU -> RMSNorm -> Linear -> [Classes=3]
        const int inFeatures = 8;
        const int hiddenFeatures = 16;
        const int numClasses = 3;
        const int batchSize = 12;

        using var l1 = new Linear(inFeatures, hiddenFeatures, seed: 101);
        using var normWeights = TensorFloatExtensions.Ones(hiddenFeatures);
        normWeights.RequiresGrad = true;
        normWeights.Grad = Tensor<float>.Zeros(hiddenFeatures);

        using var l2 = new Linear(hiddenFeatures, numClasses, seed: 202);

        var allParams = new List<Tensor<float>> { l1.Weight, l1.Bias, normWeights, l2.Weight, l2.Bias };
        using var optimizer = new AdamW(allParams, lr: 0.08f, weightDecay: 0.001f);

        // Fixed synthetic classification dataset:
        // Classes separated by feature values
        using var x = new Tensor<float>(batchSize, inFeatures);
        using var targets = new Tensor<float>(batchSize, numClasses);
        var rng = new Random(42);

        for (int b = 0; b < batchSize; b++)
        {
            int targetClass = b % numClasses;
            targets[b, targetClass] = 1.0f;

            for (int f = 0; f < inFeatures; f++)
            {
                x[b, f] = (targetClass * 2.0f) + (float)(rng.NextDouble() * 0.5);
            }
        }

        float initialLoss = 0f;
        float finalLoss = 0f;

        // Train for 150 epochs
        for (int epoch = 0; epoch < 150; epoch++)
        {
            optimizer.ZeroGrad();

            using var tape = new AutogradTape();
            foreach (var p in allParams) tape.Watch(p);

            // Forward:
            // 1. Linear 1
            using var h1 = l1.Forward(x);
            // 2. GELU activation
            using var act1 = TensorOps.GELU(h1);
            // 3. RMSNorm
            using var normed = TensorOps.RMSNorm(act1, normWeights);
            // 4. Linear 2 (Logits)
            using var logits = l2.Forward(normed);

            // 5. Cross Entropy Loss
            var (lossVal, lossTensor) = LossFunctions.CrossEntropy(logits, targets);

            if (epoch == 0) initialLoss = lossVal;
            finalLoss = lossVal;

            // Backward: walks from lossTensor through logits -> l2 -> RMSNorm -> GELU -> l1
            tape.Backward(lossTensor);

            optimizer.Step();
        }

        Assert.True(finalLoss < initialLoss * 0.1f,
            $"Training failed to converge: Initial={initialLoss:F4}, Final={finalLoss:F4} (expected >10x reduction)");
        Assert.True(finalLoss < 0.15f,
            $"Final loss should be < 0.15, was {finalLoss:F4}");
    }
}
