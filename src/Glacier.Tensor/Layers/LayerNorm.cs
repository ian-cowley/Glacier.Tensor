using System;
using System.Collections.Generic;
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
        using var inC = input.Contiguous();
        var output = new Tensor<float>(input.Shape.AsSpan());

        int d = inC.Shape[inC.Rank - 1];
        int batch = (int)(inC.ElementCount / d);

        var inSpan = inC.AsSpan();
        var outSpan = output.AsSpan();
        var gammaSpan = _gamma.AsSpan();
        var betaSpan = _beta.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            int offset = b * d;

            // Calculate mean
            float mean = 0f;
            for (int i = 0; i < d; i++) mean += inSpan[offset + i];
            mean /= d;

            // Calculate variance
            float var = 0f;
            for (int i = 0; i < d; i++)
            {
                float diff = inSpan[offset + i] - mean;
                var += diff * diff;
            }
            var /= d;
            float invStd = 1.0f / MathF.Sqrt(var + _eps);

            // Normalize and scale
            for (int i = 0; i < d; i++)
            {
                float norm = (inSpan[offset + i] - mean) * invStd;
                outSpan[offset + i] = norm * gammaSpan[i] + betaSpan[i];
            }
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
