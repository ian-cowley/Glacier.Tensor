using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Layers;

/// <summary>
/// Fully-connected dense linear neural layer: Y = X * W + B.
/// </summary>
public sealed unsafe class Linear : ILayer
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
        ArgumentNullException.ThrowIfNull(input);

        // input: [..., inFeatures], weight: [inFeatures, outFeatures] -> [..., outFeatures]
        var output = TensorOps.MatMul(input, _weight);

        int outFeatures = _weight.Shape[1];
        if (outFeatures > 0 && output.ElementCount > 0)
        {
            int batch = checked((int)(output.ElementCount / outFeatures));
            using var biasContig = (_bias.IsContiguous && _bias.ElementOffset == 0) ? null : _bias.Contiguous();
            var biasSrc = biasContig ?? _bias;
            AddBiasBroadcast(output.DataPointer, biasSrc.DataPointer, batch, outFeatures);
        }

        output.RequiresGrad |= _bias.RequiresGrad;

        // Record bias addition on tape if active
        if (AutogradTape.Current != null && _bias.RequiresGrad)
        {
            AutogradTape.Current.Watch(_bias);
            // Backward pass for bias broadcast: sum across leading batch dimensions
            AutogradTape.Current.Record(new TapeEntry(AutogradOp.Add, output, null, _bias));
        }

        return output;
    }

    internal static void AddBiasBroadcast(float* outPtr, float* biasPtr, int batch, int outFeatures)
    {
        for (int b = 0; b < batch; b++)
        {
            float* row = outPtr + (long)b * outFeatures;
            int f = 0;

            if (Vector512.IsHardwareAccelerated)
            {
                int step = Vector512<float>.Count;
                for (; f + step <= outFeatures; f += step)
                {
                    Vector512.Store(Vector512.Load(row + f) + Vector512.Load(biasPtr + f), row + f);
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                int step = Vector256<float>.Count;
                for (; f + step <= outFeatures; f += step)
                {
                    Vector256.Store(Vector256.Load(row + f) + Vector256.Load(biasPtr + f), row + f);
                }
            }

            if (Vector128.IsHardwareAccelerated)
            {
                int step = Vector128<float>.Count;
                for (; f + step <= outFeatures; f += step)
                {
                    Vector128.Store(Vector128.Load(row + f) + Vector128.Load(biasPtr + f), row + f);
                }
            }

            for (; f < outFeatures; f++)
            {
                row[f] += biasPtr[f];
            }
        }
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
