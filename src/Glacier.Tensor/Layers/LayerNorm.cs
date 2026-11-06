using System;
using System.Collections.Generic;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Layers;

/// <summary>
/// Layer Normalization across normalized shape features.
/// </summary>
public sealed class LayerNorm : ILayer
{
    private readonly Tensor<float> _gamma;
    private readonly Tensor<float> _beta;
    private readonly float _eps;
    private readonly List<Tensor<float>> _parameters;
    private readonly List<Tensor<float>> _gradients;
    private bool _disposed;

    public Tensor<float> Gamma => _gamma;
    public Tensor<float> Beta => _beta;
    public IReadOnlyList<Tensor<float>> Parameters => _parameters;
    public IReadOnlyList<Tensor<float>> Gradients => _gradients;

    public LayerNorm(int normalizedShape, float eps = 1e-5f)
    {
        _eps = eps;
        _gamma = TensorFloatExtensions.Ones(normalizedShape);
        _beta = Tensor<float>.Zeros(normalizedShape);

        _gamma.RequiresGrad = true;
        _beta.RequiresGrad = true;
        _gamma.Grad = Tensor<float>.Zeros(normalizedShape);
        _beta.Grad = Tensor<float>.Zeros(normalizedShape);

        _parameters = new List<Tensor<float>> { _gamma, _beta };
        _gradients = new List<Tensor<float>> { _gamma.Grad, _beta.Grad };
    }

    public Tensor<float> Forward(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var output = new Tensor<float>(input.Shape.AsSpan());
        ElementwiseKernels.LayerNorm(input, _gamma, _beta, output, _eps);

        output.RequiresGrad = input.RequiresGrad || _gamma.RequiresGrad || _beta.RequiresGrad;
        if (output.RequiresGrad && AutogradTape.Current is { IsExecutingBackward: false } tape)
        {
            tape.Record(new TapeEntry(AutogradOp.LayerNorm, output, input, _gamma, _eps, _beta));
        }

        return output;
    }

    public void ZeroGrad()
    {
        _gamma.Grad?.Fill(0.0f);
        _beta.Grad?.Fill(0.0f);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _gamma.Dispose();
            _beta.Dispose();
        }
    }
}
