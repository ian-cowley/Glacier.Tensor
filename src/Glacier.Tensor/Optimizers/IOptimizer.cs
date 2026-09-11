using System;

namespace Glacier.Tensor.Optimizers;

/// <summary>
/// Parameter optimization contract.
/// </summary>
public interface IOptimizer : IDisposable
{
    void Step();
    void ZeroGrad();
}
