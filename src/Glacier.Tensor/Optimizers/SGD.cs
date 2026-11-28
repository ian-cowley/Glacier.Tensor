using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
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
            UpdateParameterSimd(pSpan, gSpan, vSpan, _lr, _momentum, _weightDecay);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UpdateParameterSimd(
        Span<float> pSpan,
        ReadOnlySpan<float> gSpan,
        Span<float> vSpan,
        float lr,
        float momentum,
        float weightDecay)
    {
        int n = pSpan.Length;
        int i = 0;

        ref float rP = ref MemoryMarshal.GetReference(pSpan);
        ref float rG = ref MemoryMarshal.GetReference(gSpan);
        ref float rV = ref MemoryMarshal.GetReference(vSpan);

        if (Vector512.IsHardwareAccelerated && n >= Vector512<float>.Count)
        {
            var vLr = Vector512.Create(lr);
            var vMom = Vector512.Create(momentum);
            var vWd = Vector512.Create(weightDecay);
            int step = Vector512<float>.Count;
            int limit = n - step;
            for (; i <= limit; i += step)
            {
                var vp = Vector512.LoadUnsafe(ref rP, (nuint)i);
                var vg = Vector512.LoadUnsafe(ref rG, (nuint)i);
                var vv = Vector512.LoadUnsafe(ref rV, (nuint)i);

                var g = vg + vWd * vp;
                vv = vMom * vv + g;
                vp -= vLr * vv;

                vp.StoreUnsafe(ref rP, (nuint)i);
                vv.StoreUnsafe(ref rV, (nuint)i);
            }
        }

        if (Vector256.IsHardwareAccelerated && n - i >= Vector256<float>.Count)
        {
            var vLr = Vector256.Create(lr);
            var vMom = Vector256.Create(momentum);
            var vWd = Vector256.Create(weightDecay);
            int step = Vector256<float>.Count;
            int limit = n - step;
            for (; i <= limit; i += step)
            {
                var vp = Vector256.LoadUnsafe(ref rP, (nuint)i);
                var vg = Vector256.LoadUnsafe(ref rG, (nuint)i);
                var vv = Vector256.LoadUnsafe(ref rV, (nuint)i);

                var g = vg + vWd * vp;
                vv = vMom * vv + g;
                vp -= vLr * vv;

                vp.StoreUnsafe(ref rP, (nuint)i);
                vv.StoreUnsafe(ref rV, (nuint)i);
            }
        }

        if (Vector128.IsHardwareAccelerated && n - i >= Vector128<float>.Count)
        {
            var vLr = Vector128.Create(lr);
            var vMom = Vector128.Create(momentum);
            var vWd = Vector128.Create(weightDecay);
            int step = Vector128<float>.Count;
            int limit = n - step;
            for (; i <= limit; i += step)
            {
                var vp = Vector128.LoadUnsafe(ref rP, (nuint)i);
                var vg = Vector128.LoadUnsafe(ref rG, (nuint)i);
                var vv = Vector128.LoadUnsafe(ref rV, (nuint)i);

                var g = vg + vWd * vp;
                vv = vMom * vv + g;
                vp -= vLr * vv;

                vp.StoreUnsafe(ref rP, (nuint)i);
                vv.StoreUnsafe(ref rV, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            float g = gSpan[i] + weightDecay * pSpan[i];
            vSpan[i] = momentum * vSpan[i] + g;
            pSpan[i] -= lr * vSpan[i];
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
