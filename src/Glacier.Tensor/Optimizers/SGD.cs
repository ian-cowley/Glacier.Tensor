using System;
using System.Collections.Generic;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Optimizers;

/// <summary>
/// Stochastic Gradient Descent (SGD) with momentum.
/// </summary>
public sealed class SGD : IOptimizer
{
    private readonly List<Tensor<float>> _parameters = new();
    private readonly List<Tensor<float>> _velocity = new();
    private readonly float _lr;
    private readonly float _momentum;
    private readonly float _weightDecay;
    private bool _disposed;

    public float LearningRate => _lr;

    public SGD(
        IEnumerable<Tensor<float>> parameters,
        float lr = 1e-2f,
        float momentum = 0.9f,
        float weightDecay = 0.0f)
    {
        _lr = lr;
        _momentum = momentum;
        _weightDecay = weightDecay;

        foreach (var p in parameters)
        {
            _parameters.Add(p);
            _velocity.Add(Tensor<float>.Zeros(p.Shape.AsSpan()));
        }
    }

    public void Step()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        for (int pIdx = 0; pIdx < _parameters.Count; pIdx++)
        {
            var p = _parameters[pIdx];
            var grad = p.Grad;
            if (grad == null) continue;

            var v = _velocity[pIdx];

            var pSpan = p.AsSpan();
            var gSpan = grad.AsSpan();
            var vSpan = v.AsSpan();
            int n = pSpan.Length;

            for (int i = 0; i < n; i++)
            {
                float g = gSpan[i] + _weightDecay * pSpan[i];
                vSpan[i] = _momentum * vSpan[i] + g;
                pSpan[i] -= _lr * vSpan[i];
            }
        }
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters)
        {
            p.Grad?.Fill(0.0f);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            foreach (var t in _velocity) t.Dispose();
        }
    }
}
