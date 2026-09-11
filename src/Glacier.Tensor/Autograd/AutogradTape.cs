using System;
using System.Collections.Generic;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Autograd;

/// <summary>
/// Zero-allocation reverse-mode automatic differentiation tape.
/// Records operations on the forward pass and executes reverse-mode gradient propagation.
/// </summary>
public sealed class AutogradTape : IDisposable
{
    [ThreadStatic]
    private static AutogradTape? s_current;

    public static AutogradTape? Current => s_current;

    private readonly List<TapeEntry> _entries = new(256);
    private bool _disposed;

    public AutogradTape()
    {
        s_current = this;
    }

    public void Watch(Tensor<float> tensor)
    {
        tensor.RequiresGrad = true;
        if (tensor.Grad == null)
        {
            tensor.Grad = new Tensor<float>(tensor.Shape);
        }
    }

    public void Record(TapeEntry entry)
    {
        _entries.Add(entry);
    }

    public void Backward(Tensor<float> output)
    {
        // Seed output gradient with 1.0f if not already seeded
        if (output.Grad == null)
        {
            output.Grad = new Tensor<float>(output.Shape);
            output.Grad.Fill(1.0f);
        }

        // Reverse topological traversal over tape entries
        for (int i = _entries.Count - 1; i >= 0; i--)
        {
            var entry = _entries[i];
            var dY = entry.Output.Grad;
            if (dY == null) continue;

            switch (entry.Op)
            {
                case AutogradOp.Add:
                {
                    if (entry.Input0?.RequiresGrad == true)
                        AccumulateGradient(entry.Input0, dY);
                    if (entry.Input1?.RequiresGrad == true)
                        AccumulateGradient(entry.Input1, dY);
                    break;
                }

                case AutogradOp.Subtract:
                {
                    if (entry.Input0?.RequiresGrad == true)
                        AccumulateGradient(entry.Input0, dY);
                    if (entry.Input1?.RequiresGrad == true)
                    {
                        using var negDy = new Tensor<float>(dY.Shape);
                        ElementwiseKernels.Scale(dY, -1.0f, negDy);
                        AccumulateGradient(entry.Input1, negDy);
                    }
                    break;
                }

                case AutogradOp.Multiply:
                {
                    if (entry.Input0?.RequiresGrad == true && entry.Input1 != null)
                    {
                        using var da = new Tensor<float>(entry.Input0.Shape);
                        ElementwiseKernels.Multiply(dY, entry.Input1, da);
                        AccumulateGradient(entry.Input0, da);
                    }
                    if (entry.Input1?.RequiresGrad == true && entry.Input0 != null)
                    {
                        using var db = new Tensor<float>(entry.Input1.Shape);
                        ElementwiseKernels.Multiply(dY, entry.Input0, db);
                        AccumulateGradient(entry.Input1, db);
                    }
                    break;
                }

                case AutogradOp.Scale:
                {
                    if (entry.Input0?.RequiresGrad == true)
                    {
                        using var da = new Tensor<float>(entry.Input0.Shape);
                        ElementwiseKernels.Scale(dY, entry.Scalar, da);
                        AccumulateGradient(entry.Input0, da);
                    }
                    break;
                }

                case AutogradOp.MatMul:
                {
                    var a = entry.Input0!;
                    var b = entry.Input1!;

                    // dY is [M, N]
                    // dA = dY x B^T  ([M, N] x [N, K] -> [M, K])
                    if (a.RequiresGrad)
                    {
                        using var bT = b.Transpose();
                        using var da = new Tensor<float>(a.Shape);
                        GemmKernels.MatMul(dY, bT, da);
                        AccumulateGradient(a, da);
                    }

                    // dB = A^T x dY  ([K, M] x [M, N] -> [K, N])
                    if (b.RequiresGrad)
                    {
                        using var aT = a.Transpose();
                        using var db = new Tensor<float>(b.Shape);
                        GemmKernels.MatMul(aT, dY, db);
                        AccumulateGradient(b, db);
                    }
                    break;
                }

                case AutogradOp.ReLU:
                {
                    if (entry.Input0?.RequiresGrad == true)
                    {
                        using var dx = new Tensor<float>(entry.Input0.Shape);
                        ElementwiseKernels.ReLUBackward(dY, entry.Input0, dx);
                        AccumulateGradient(entry.Input0, dx);
                    }
                    break;
                }

                case AutogradOp.Sigmoid:
                {
                    if (entry.Input0?.RequiresGrad == true)
                    {
                        // dx = dy * y * (1 - y)
                        using var y = entry.Output;
                        using var oneMinusY = new Tensor<float>(y.Shape);
                        oneMinusY.Fill(1.0f);
                        ElementwiseKernels.Subtract(oneMinusY, y, oneMinusY);

                        using var dyTimesY = new Tensor<float>(y.Shape);
                        ElementwiseKernels.Multiply(dY, y, dyTimesY);

                        using var dx = new Tensor<float>(y.Shape);
                        ElementwiseKernels.Multiply(dyTimesY, oneMinusY, dx);
                        AccumulateGradient(entry.Input0, dx);
                    }
                    break;
                }
            }
        }
    }

    private static void AccumulateGradient(Tensor<float> target, Tensor<float> gradDelta)
    {
        if (target.Grad == null)
        {
            target.Grad = new Tensor<float>(target.Shape);
        }

        var spanTarget = target.Grad.AsSpan();
        var spanDelta = gradDelta.AsSpan();

        if (spanTarget.Length == spanDelta.Length)
        {
            for (int i = 0; i < spanTarget.Length; i++)
            {
                spanTarget[i] += spanDelta[i];
            }
        }
        else if (spanDelta.Length % spanTarget.Length == 0)
        {
            // Broadcast reduction (e.g. bias gradient: sum over batch rows)
            int d = spanTarget.Length;
            int batch = spanDelta.Length / d;
            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < d; i++)
                {
                    spanTarget[i] += spanDelta[b * d + i];
                }
            }
        }
    }

    public void Reset()
    {
        _entries.Clear();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _entries.Clear();
            if (ReferenceEquals(s_current, this))
            {
                s_current = null;
            }
        }
    }
}
