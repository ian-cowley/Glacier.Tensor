using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
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
    private float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private int _stepCount;
    private bool _disposed;

    public float LearningRate { get => _lr; set => _lr = value; }

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
            UpdateParameterSimd(pSpan, gSpan, mSpan, vSpan, _lr * _weightDecay, _beta1, _beta2, alpha_t, _eps);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UpdateParameterSimd(
        Span<float> pSpan,
        ReadOnlySpan<float> gSpan,
        Span<float> mSpan,
        Span<float> vSpan,
        float decayStep,
        float beta1,
        float beta2,
        float alphaT,
        float eps)
    {
        int n = pSpan.Length;
        int i = 0;
        float oneMinusBeta1 = 1.0f - beta1;
        float oneMinusBeta2 = 1.0f - beta2;

        ref float rP = ref MemoryMarshal.GetReference(pSpan);
        ref float rG = ref MemoryMarshal.GetReference(gSpan);
        ref float rM = ref MemoryMarshal.GetReference(mSpan);
        ref float rV = ref MemoryMarshal.GetReference(vSpan);

        if (Vector512.IsHardwareAccelerated && n >= Vector512<float>.Count)
        {
            var vDecay = Vector512.Create(1.0f - decayStep);
            var vB1 = Vector512.Create(beta1);
            var v1mB1 = Vector512.Create(oneMinusBeta1);
            var vB2 = Vector512.Create(beta2);
            var v1mB2 = Vector512.Create(oneMinusBeta2);
            var vAlpha = Vector512.Create(alphaT);
            var vEps = Vector512.Create(eps);

            int step = Vector512<float>.Count;
            int limit = n - step;
            for (; i <= limit; i += step)
            {
                var vp = Vector512.LoadUnsafe(ref rP, (nuint)i);
                var vg = Vector512.LoadUnsafe(ref rG, (nuint)i);
                var vm = Vector512.LoadUnsafe(ref rM, (nuint)i);
                var vv = Vector512.LoadUnsafe(ref rV, (nuint)i);

                vp *= vDecay;
                vm = vB1 * vm + v1mB1 * vg;
                vv = vB2 * vv + v1mB2 * (vg * vg);
                vp -= vAlpha * vm / (Vector512.Sqrt(vv) + vEps);

                vp.StoreUnsafe(ref rP, (nuint)i);
                vm.StoreUnsafe(ref rM, (nuint)i);
                vv.StoreUnsafe(ref rV, (nuint)i);
            }
        }

        if (Vector256.IsHardwareAccelerated && n - i >= Vector256<float>.Count)
        {
            var vDecay = Vector256.Create(1.0f - decayStep);
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(oneMinusBeta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(oneMinusBeta2);
            var vAlpha = Vector256.Create(alphaT);
            var vEps = Vector256.Create(eps);

            int step = Vector256<float>.Count;
            int limit = n - step;
            for (; i <= limit; i += step)
            {
                var vp = Vector256.LoadUnsafe(ref rP, (nuint)i);
                var vg = Vector256.LoadUnsafe(ref rG, (nuint)i);
                var vm = Vector256.LoadUnsafe(ref rM, (nuint)i);
                var vv = Vector256.LoadUnsafe(ref rV, (nuint)i);

                vp *= vDecay;
                vm = vB1 * vm + v1mB1 * vg;
                vv = vB2 * vv + v1mB2 * (vg * vg);
                vp -= vAlpha * vm / (Vector256.Sqrt(vv) + vEps);

                vp.StoreUnsafe(ref rP, (nuint)i);
                vm.StoreUnsafe(ref rM, (nuint)i);
                vv.StoreUnsafe(ref rV, (nuint)i);
            }
        }

        if (Vector128.IsHardwareAccelerated && n - i >= Vector128<float>.Count)
        {
            var vDecay = Vector128.Create(1.0f - decayStep);
            var vB1 = Vector128.Create(beta1);
            var v1mB1 = Vector128.Create(oneMinusBeta1);
            var vB2 = Vector128.Create(beta2);
            var v1mB2 = Vector128.Create(oneMinusBeta2);
            var vAlpha = Vector128.Create(alphaT);
            var vEps = Vector128.Create(eps);

            int step = Vector128<float>.Count;
            int limit = n - step;
            for (; i <= limit; i += step)
            {
                var vp = Vector128.LoadUnsafe(ref rP, (nuint)i);
                var vg = Vector128.LoadUnsafe(ref rG, (nuint)i);
                var vm = Vector128.LoadUnsafe(ref rM, (nuint)i);
                var vv = Vector128.LoadUnsafe(ref rV, (nuint)i);

                vp *= vDecay;
                vm = vB1 * vm + v1mB1 * vg;
                vv = vB2 * vv + v1mB2 * (vg * vg);
                vp -= vAlpha * vm / (Vector128.Sqrt(vv) + vEps);

                vp.StoreUnsafe(ref rP, (nuint)i);
                vm.StoreUnsafe(ref rM, (nuint)i);
                vv.StoreUnsafe(ref rV, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            float g = gSpan[i];
            pSpan[i] -= decayStep * pSpan[i];
            mSpan[i] = beta1 * mSpan[i] + oneMinusBeta1 * g;
            vSpan[i] = beta2 * vSpan[i] + oneMinusBeta2 * (g * g);
            pSpan[i] -= alphaT * mSpan[i] / (MathF.Sqrt(vSpan[i]) + eps);
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
