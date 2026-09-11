using System;
using System.Collections.Generic;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Layers;

/// <summary>
/// Core neural network layer interface.
/// </summary>
public interface ILayer : IDisposable
{
    Tensor<float> Forward(Tensor<float> input);
    void ZeroGrad();
    IReadOnlyList<Tensor<float>> Parameters { get; }
    IReadOnlyList<Tensor<float>> Gradients { get; }
}
