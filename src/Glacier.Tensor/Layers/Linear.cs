using System;
using System.Collections.Generic;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Layers;

/// <summary>
/// Fully-connected dense linear neural layer: Y = X * W + B.
/// </summary>
public sealed class Linear : ILayer
{
    private readonly Tensor<float> _weight;
    private readonly Tensor<float> _bias;
    private readonly List<Tensor<float>> _parameters;
    private readonly List<Tensor<float>> _gradients;
    private bool _disposed;

    public Tensor<float> Weight => _weight;
    public Tensor<float> Bias => _bias;
    public IReadOnlyList<Tensor<float>> Parameters => _parameters;
    public IReadOnlyList<Tensor<float>> Gradients => _gradients;

    public Linear(int inFeatures, int outFeatures, int? seed = null)
    {
        _weight = TensorFloatExtensions.KaimingUniform([inFeatures, outFeatures], inFeatures, seed);
        _bias = Tensor<float>.Zeros(1, outFeatures);

        _weight.RequiresGrad = true;
        _bias.RequiresGrad = true;
        _weight.Grad = Tensor<float>.Zeros(inFeatures, outFeatures);
        _bias.Grad = Tensor<float>.Zeros(1, outFeatures);

        _parameters = new List<Tensor<float>> { _weight, _bias };
        _gradients = new List<Tensor<float>> { _weight.Grad, _bias.Grad };
    }

    public Tensor<float> Forward(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // input: [Batch, inFeatures], weight: [inFeatures, outFeatures] -> [Batch, outFeatures]
        var output = TensorOps.MatMul(input, _weight);

        // Add bias broadcasted across batch rows
        int batch = input.Shape[0];
        int outFeatures = _weight.Shape[1];
        var outSpan = output.AsSpan();
        var biasSpan = _bias.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < outFeatures; f++)
            {
                outSpan[b * outFeatures + f] += biasSpan[f];
            }
        }

        // Record bias addition on tape if active
        if (AutogradTape.Current != null)
        {
            AutogradTape.Current.Watch(_bias);
            // Backward pass for bias broadcast: sum across batch dimension
            AutogradTape.Current.Record(new TapeEntry(AutogradOp.Add, output, null, _bias));
        }

        return output;
    }

    public void ZeroGrad()
    {
        _weight.Grad?.Fill(0.0f);
        _bias.Grad?.Fill(0.0f);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weight.Dispose();
            _bias.Dispose();
        }
    }
}
