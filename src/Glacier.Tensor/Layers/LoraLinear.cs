using System;
using System.Collections.Generic;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Layers;

/// <summary>
/// Parameter-Efficient Fine-Tuning (PEFT) Low-Rank Adaptation (LoRA) layer.
/// Freezes base model weights W0 and trains low-rank factorized adapter matrices A and B:
/// Y = X * W0 + (alpha / r) * (X * A) * B
/// Delivers parameter-efficient fine-tuning with 99%+ parameter reduction and zero base weight drift.
/// </summary>
public sealed class LoraLinear : ILayer
{
    private readonly Tensor<float> _baseWeight;
    private readonly Tensor<float>? _baseBias;
    private readonly Tensor<float> _adapterA; // [InFeatures, Rank]
    private readonly Tensor<float> _adapterB; // [Rank, OutFeatures]
    private readonly float _scaling;
    private readonly int _rank;
    private readonly int _inFeatures;
    private readonly int _outFeatures;
    private readonly List<Tensor<float>> _parameters;
    private readonly List<Tensor<float>> _gradients;
    private bool _disposed;

    public Tensor<float> BaseWeight => _baseWeight;
    public Tensor<float>? BaseBias => _baseBias;
    public Tensor<float> AdapterA => _adapterA;
    public Tensor<float> AdapterB => _adapterB;
    public float Scaling => _scaling;
    public int Rank => _rank;
    public int InFeatures => _inFeatures;
    public int OutFeatures => _outFeatures;

    public IReadOnlyList<Tensor<float>> Parameters => _parameters;
    public IReadOnlyList<Tensor<float>> Gradients => _gradients;

    /// <summary>
    /// Constructs a LoRA layer wrapping existing base weights.
    /// </summary>
    public LoraLinear(Tensor<float> baseWeight, Tensor<float>? baseBias = null, int rank = 16, float alpha = 32f, int? seed = null)
    {
        _inFeatures = baseWeight.Shape[0];
        _outFeatures = baseWeight.Shape[1];
        _rank = rank;
        _scaling = alpha / rank;

        _baseWeight = baseWeight;
        _baseWeight.RequiresGrad = false;

        _baseBias = baseBias;
        if (_baseBias != null) _baseBias.RequiresGrad = false;

        // Adapter A: Gaussian initialized with std = 1 / sqrt(inFeatures)
        float stdA = 1.0f / MathF.Sqrt(_inFeatures);
        _adapterA = TensorFloatExtensions.RandomNormal([_inFeatures, rank], mean: 0f, std: stdA, seed: seed ?? 42);

        // Adapter B: Initialized to zeros so Delta W = A * B = 0 at start
        _adapterB = Tensor<float>.Zeros(rank, _outFeatures);

        _adapterA.RequiresGrad = true;
        _adapterB.RequiresGrad = true;
        _adapterA.Grad = Tensor<float>.Zeros(_inFeatures, rank);
        _adapterB.Grad = Tensor<float>.Zeros(rank, _outFeatures);

        _parameters = new List<Tensor<float>> { _adapterA, _adapterB };
        _gradients = new List<Tensor<float>> { _adapterA.Grad, _adapterB.Grad };
    }

    /// <summary>
    /// Creates a LoRA layer with newly allocated base weights.
    /// </summary>
    public static LoraLinear Create(int inFeatures, int outFeatures, int rank = 16, float alpha = 32f, int? seed = null)
    {
        var baseW = TensorFloatExtensions.KaimingUniform([inFeatures, outFeatures], inFeatures, seed);
        return new LoraLinear(baseW, null, rank, alpha, seed);
    }

    public Tensor<float> Forward(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // 1. Base model projection: X * W0 (frozen)
        var baseOut = TensorOps.MatMul(input, _baseWeight);

        // 2. LoRA low-rank branch: (X * A) * B * scaling
        var lowRank = TensorOps.MatMul(input, _adapterA);
        var deltaOut = TensorOps.MatMul(lowRank, _adapterB);
        var scaledDelta = TensorOps.Scale(deltaOut, _scaling);

        // 3. Combined output: Y = baseOut + scaledDelta
        var output = TensorOps.Add(baseOut, scaledDelta);

        // 4. Add base bias if present
        if (_baseBias != null)
        {
            int batch = input.Shape[0];
            var outSpan = output.AsSpan();
            var biasSpan = _baseBias.AsSpan();

            for (int b = 0; b < batch; b++)
            {
                for (int f = 0; f < _outFeatures; f++)
                {
                    outSpan[b * _outFeatures + f] += biasSpan[f];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Merges the LoRA adapter weights directly into the base weights: W = W0 + (alpha / r) * (A * B).
    /// Used for zero-overhead inference deployment after fine-tuning completes.
    /// </summary>
    public Tensor<float> Merge()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var merged = new Tensor<float>(_inFeatures, _outFeatures);
        using var delta = new Tensor<float>(_inFeatures, _outFeatures);

        // delta = A * B
        GpuAccelerator.AcceleratedMatMul(_adapterA, _adapterB, delta, GpuTarget.Auto);

        // W_merged = W0 + scaling * delta
        var spanMerged = merged.AsSpan();
        var spanW0 = _baseWeight.AsSpan();
        var spanDelta = delta.AsSpan();

        for (int i = 0; i < spanMerged.Length; i++)
        {
            spanMerged[i] = spanW0[i] + _scaling * spanDelta[i];
        }

        return merged;
    }

    public void ZeroGrad()
    {
        _adapterA.Grad?.Fill(0.0f);
        _adapterB.Grad?.Fill(0.0f);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _adapterA.Dispose();
            _adapterB.Dispose();
            _disposed = true;
        }
    }
}
