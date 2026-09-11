using System;
using System.Collections.Generic;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Optimizers;

/// <summary>
/// AdamW (Adam with decoupled weight decay) optimizer.
/// State-of-the-art neural network optimizer with bias-corrected momentum and variance.
/// </summary>
public sealed class AdamW : IOptimizer
{
    private readonly List<Tensor<float>> _parameters = new();
    private readonly List<Tensor<float>> _m = new();
    private readonly List<Tensor<float>> _v = new();
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private int _stepCount;
    private bool _disposed;

    public float LearningRate => _lr;

    public AdamW(
        IEnumerable<Tensor<float>> parameters,
        float lr = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.01f)
    {
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;

        foreach (var p in parameters)
        {
            _parameters.Add(p);
            _m.Add(Tensor<float>.Zeros(p.Shape.AsSpan()));
            _v.Add(Tensor<float>.Zeros(p.Shape.AsSpan()));
        }
    }

    public void Step()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _stepCount++;

        float beta1_t = MathF.Pow(_beta1, _stepCount);
        float beta2_t = MathF.Pow(_beta2, _stepCount);
        float alpha_t = _lr * MathF.Sqrt(1.0f - beta2_t) / (1.0f - beta1_t);

        for (int pIdx = 0; pIdx < _parameters.Count; pIdx++)
        {
            var p = _parameters[pIdx];
            var grad = p.Grad;
            if (grad == null) continue;

            var m = _m[pIdx];
            var v = _v[pIdx];

            var pSpan = p.AsSpan();
            var gSpan = grad.AsSpan();
            var mSpan = m.AsSpan();
            var vSpan = v.AsSpan();
            int n = pSpan.Length;

            for (int i = 0; i < n; i++)
            {
                float g = gSpan[i];

                // Decoupled weight decay
                pSpan[i] -= _lr * _weightDecay * pSpan[i];

                // Update biased 1st moment estimate
                mSpan[i] = _beta1 * mSpan[i] + (1.0f - _beta1) * g;

                // Update biased 2nd raw moment estimate
                vSpan[i] = _beta2 * vSpan[i] + (1.0f - _beta2) * (g * g);

                // Update parameters with bias-corrected step
                pSpan[i] -= alpha_t * mSpan[i] / (MathF.Sqrt(vSpan[i]) + _eps);
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
            foreach (var t in _m) t.Dispose();
            foreach (var t in _v) t.Dispose();
        }
    }
}
