using System;
using System.Collections.Generic;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Layers;

/// <summary>
/// Sequential neural layer pipeline composing multiple layers end-to-end.
/// </summary>
public sealed class Sequential : ILayer
{
    private readonly List<ILayer> _layers;
    private readonly List<Tensor<float>> _parameters = new();
    private readonly List<Tensor<float>> _gradients = new();
    private bool _disposed;

    public IReadOnlyList<ILayer> Layers => _layers;
    public IReadOnlyList<Tensor<float>> Parameters => _parameters;
    public IReadOnlyList<Tensor<float>> Gradients => _gradients;

    public Sequential(params ILayer[] layers)
    {
        _layers = new List<ILayer>(layers);
        foreach (var layer in _layers)
        {
            _parameters.AddRange(layer.Parameters);
            _gradients.AddRange(layer.Gradients);
        }
    }

    public Tensor<float> Forward(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var current = input;
        for (int i = 0; i < _layers.Count; i++)
        {
            current = _layers[i].Forward(current);
        }
        return current;
    }

    public void ZeroGrad()
    {
        foreach (var layer in _layers)
        {
            layer.ZeroGrad();
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            foreach (var layer in _layers)
            {
                layer.Dispose();
            }
        }
    }
}
