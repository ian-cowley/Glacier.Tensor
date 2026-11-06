using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
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
    private static Stack<AutogradTape>? t_tapeStack;

    public static AutogradTape? Current => t_tapeStack is { Count: > 0 } stack ? stack.Peek() : null;

    internal bool IsExecutingBackward { get; private set; }

    private readonly List<TapeEntry> _entries = new(256);
    private bool _disposed;

    public AutogradTape()
    {
        (t_tapeStack ??= new Stack<AutogradTape>(4)).Push(this);
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
        if (!IsExecutingBackward)
        {
            _entries.Add(entry);
        }
    }

    public void Backward(Tensor<float> output, bool retainGraph = false)
    {
        // Seed output gradient with 1.0f if not already seeded
        if (output.Grad == null)
        {
            output.Grad = new Tensor<float>(output.Shape);
            output.Grad.Fill(1.0f);
        }

        bool prevBackward = IsExecutingBackward;
        IsExecutingBackward = true;
        try
        {
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
                            var y = entry.Output;
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

                    case AutogradOp.GELU:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = new Tensor<float>(entry.Input0.Shape);
                            ElementwiseKernels.GELUBackward(dY, entry.Input0, dx);
                            AccumulateGradient(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.Softmax:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = new Tensor<float>(entry.Input0.Shape);
                            ElementwiseKernels.SoftmaxBackward(dY, entry.Output, dx, entry.IntParam0);
                            AccumulateGradient(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.RMSNorm:
                    {
                        var a = entry.Input0;
                        var w = entry.Input1;
                        float eps = entry.Scalar > 0 ? entry.Scalar : 1e-5f;

                        Tensor<float>? da = a?.RequiresGrad == true ? new Tensor<float>(a.Shape) : null;
                        Tensor<float>? dw = w?.RequiresGrad == true ? new Tensor<float>(w.Shape) : null;

                        if (da != null || dw != null)
                        {
                            using var scopedDa = da;
                            using var scopedDw = dw;
                            using var dummyDa = da == null ? new Tensor<float>(a!.Shape) : null;

                            ElementwiseKernels.RMSNormBackward(dY, a!, w, da ?? dummyDa!, dw, eps);

                            if (da != null)
                            {
                                AccumulateGradient(a!, da);
                            }
                            if (dw != null)
                            {
                                AccumulateGradient(w!, dw);
                            }
                        }
                        break;
                    }

                    case AutogradOp.LayerNorm:
                    {
                        var a = entry.Input0;
                        var gamma = entry.Input1;
                        var beta = entry.Input2;
                        float eps = entry.Scalar > 0 ? entry.Scalar : 1e-5f;

                        Tensor<float>? da = a?.RequiresGrad == true ? new Tensor<float>(a.Shape) : null;
                        Tensor<float>? dGamma = gamma?.RequiresGrad == true ? new Tensor<float>(gamma.Shape) : null;
                        Tensor<float>? dBeta = beta?.RequiresGrad == true ? new Tensor<float>(beta.Shape) : null;

                        if (da != null || dGamma != null || dBeta != null)
                        {
                            using var scopedDa = da;
                            using var scopedDGamma = dGamma;
                            using var scopedDBeta = dBeta;
                            using var dummyDa = da == null ? new Tensor<float>(a!.Shape) : null;

                            ElementwiseKernels.LayerNormBackward(dY, a!, gamma, da ?? dummyDa!, dGamma, dBeta, eps);

                            if (da != null) AccumulateGradient(a!, da);
                            if (dGamma != null) AccumulateGradient(gamma!, dGamma);
                            if (dBeta != null) AccumulateGradient(beta!, dBeta);
                        }
                        break;
                    }

                    case AutogradOp.Transpose:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = dY.Transpose(entry.IntParam0, entry.IntParam1);
                            AccumulateGradient(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.Reshape:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = dY.Reshape(entry.Input0.Shape.AsSpan());
                            AccumulateGradient(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.Slice:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            entry.Input0.Grad ??= new Tensor<float>(entry.Input0.Shape);
                            using var gradSlice = entry.Input0.Grad.Slice(entry.IntParam0, entry.IntParam1, entry.IntParam2);
                            AccumulateIntoTensor(gradSlice, dY);
                        }
                        break;
                    }

                    case AutogradOp.Contiguous:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            AccumulateGradient(entry.Input0, dY);
                        }
                        break;
                    }

                    case AutogradOp.MSELoss:
                    {
                        var pred = entry.Input0;
                        var targets = entry.Input1;
                        if (pred?.RequiresGrad == true && targets != null)
                        {
                            float dLoss = GetScalar(dY);
                            using var dPred = new Tensor<float>(pred.Shape);
                            Losses.LossFunctions.ComputeMSEGradient(pred, targets, dPred, dLoss);
                            pred.Grad?.Fill(0.0f);
                            AccumulateGradient(pred, dPred);
                        }
                        break;
                    }

                    case AutogradOp.BinaryCrossEntropyLoss:
                    {
                        var pred = entry.Input0;
                        var targets = entry.Input1;
                        if (pred?.RequiresGrad == true && targets != null)
                        {
                            float dLoss = GetScalar(dY);
                            using var dPred = new Tensor<float>(pred.Shape);
                            Losses.LossFunctions.ComputeBinaryCrossEntropyGradient(pred, targets, dPred, dLoss);
                            pred.Grad?.Fill(0.0f);
                            AccumulateGradient(pred, dPred);
                        }
                        break;
                    }

                    case AutogradOp.CrossEntropyLoss:
                    {
                        var logits = entry.Input0;
                        var targets = entry.Input1;
                        if (logits?.RequiresGrad == true && targets != null)
                        {
                            float dLoss = GetScalar(dY);
                            using var dLogits = new Tensor<float>(logits.Shape);
                            Losses.LossFunctions.ComputeCrossEntropyGradient(logits, targets, dLogits, dLoss);
                            logits.Grad?.Fill(0.0f);
                            AccumulateGradient(logits, dLogits);
                        }
                        break;
                    }

                    case AutogradOp.CategoricalCrossEntropyLoss:
                    {
                        var logits = entry.Input0;
                        var targets = entry.Input1;
                        if (logits?.RequiresGrad == true && targets != null)
                        {
                            float dLoss = GetScalar(dY);
                            using var dLogits = new Tensor<float>(logits.Shape);
                            Losses.LossFunctions.ComputeCategoricalCrossEntropyGradient(logits, targets, dLogits, entry.Scalar, dLoss);
                            logits.Grad?.Fill(0.0f);
                            AccumulateGradient(logits, dLogits);
                        }
                        break;
                    }

                    case AutogradOp.BrierScoreLoss:
                    {
                        var logits = entry.Input0;
                        var targets = entry.Input1;
                        if (logits?.RequiresGrad == true && targets != null)
                        {
                            float dLoss = GetScalar(dY);
                            using var dLogits = new Tensor<float>(logits.Shape);
                            Losses.LossFunctions.ComputeBrierScoreGradient(logits, targets, dLogits, dLoss);
                            logits.Grad?.Fill(0.0f);
                            AccumulateGradient(logits, dLogits);
                        }
                        break;
                    }
                }
            }
        }
        finally
        {
            IsExecutingBackward = prevBackward;
        }

        if (!retainGraph)
        {
            _entries.Clear();
        }
    }

    private static float GetScalar(Tensor<float> scalarTensor)
    {
        if (scalarTensor.IsContiguous)
        {
            return scalarTensor.AsSpan()[0];
        }
        Span<int> zeros = stackalloc int[scalarTensor.Rank];
        return scalarTensor[zeros];
    }

    private static void AccumulateGradient(Tensor<float> target, Tensor<float> gradDelta)
    {
        if (target.Grad == null)
        {
            target.Grad = new Tensor<float>(target.Shape);
        }

        if (!target.Grad.IsContiguous)
        {
            AccumulateIntoTensor(target.Grad, gradDelta);
            return;
        }

        using var deltaContig = gradDelta.IsContiguous ? null : gradDelta.Contiguous();
        var spanTarget = target.Grad.AsSpan();
        var spanDelta = (deltaContig ?? gradDelta).AsSpan();

        if (spanTarget.Length == spanDelta.Length)
        {
            AddInPlaceVectorized(spanTarget, spanDelta);
        }
        else if (spanDelta.Length % spanTarget.Length == 0)
        {
            // Broadcast reduction (e.g. bias gradient: sum over batch rows)
            int d = spanTarget.Length;
            int batch = spanDelta.Length / d;
            for (int b = 0; b < batch; b++)
            {
                AddInPlaceVectorized(spanTarget, spanDelta.Slice(b * d, d));
            }
        }
    }

    private static void AccumulateIntoTensor(Tensor<float> dst, Tensor<float> delta)
    {
        if (dst.IsContiguous && delta.IsContiguous && dst.ElementCount == delta.ElementCount)
        {
            AddInPlaceVectorized(dst.AsSpan(), delta.AsSpan());
            return;
        }

        using var deltaContig = delta.IsContiguous ? null : delta.Contiguous();
        var srcSpan = (deltaContig ?? delta).AsSpan();

        int rank = dst.Rank;
        if (rank == 1)
        {
            int n = dst.Shape[0];
            for (int i = 0; i < n; i++) dst[i] += srcSpan[i];
        }
        else if (rank == 2)
        {
            int rows = dst.Shape[0];
            int cols = dst.Shape[1];
            int idx = 0;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    dst[r, c] += srcSpan[idx++];
                }
            }
        }
        else
        {
            Span<int> indices = stackalloc int[rank];
            int count = (int)dst.ElementCount;
            for (int i = 0; i < count; i++)
            {
                dst[indices] += srcSpan[i];
                for (int d = rank - 1; d >= 0; d--)
                {
                    indices[d]++;
                    if (indices[d] < dst.Shape[d]) break;
                    indices[d] = 0;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddInPlaceVectorized(Span<float> target, ReadOnlySpan<float> delta)
    {
        int n = target.Length;
        int i = 0;

        ref float rTarget = ref MemoryMarshal.GetReference(target);
        ref float rDelta = ref MemoryMarshal.GetReference(delta);

        // Stage 1: AVX-512 (16 floats per iteration)
        if (Vector512.IsHardwareAccelerated)
        {
            int step512 = Vector512<float>.Count;
            int limit512 = n - step512;
            for (; i <= limit512; i += step512)
            {
                var vTarget = Vector512.LoadUnsafe(ref rTarget, (nuint)i);
                var vDelta = Vector512.LoadUnsafe(ref rDelta, (nuint)i);
                (vTarget + vDelta).StoreUnsafe(ref rTarget, (nuint)i);
            }
        }

        // Stage 2: AVX2 (8 floats per iteration)
        if (Vector256.IsHardwareAccelerated)
        {
            int step256 = Vector256<float>.Count;
            int limit256 = n - step256;
            for (; i <= limit256; i += step256)
            {
                var vTarget = Vector256.LoadUnsafe(ref rTarget, (nuint)i);
                var vDelta = Vector256.LoadUnsafe(ref rDelta, (nuint)i);
                (vTarget + vDelta).StoreUnsafe(ref rTarget, (nuint)i);
            }
        }

        // Stage 3: ARM Neon / SSE2 (4 floats per iteration)
        if (Vector128.IsHardwareAccelerated)
        {
            int step128 = Vector128<float>.Count;
            int limit128 = n - step128;
            for (; i <= limit128; i += step128)
            {
                var vTarget = Vector128.LoadUnsafe(ref rTarget, (nuint)i);
                var vDelta = Vector128.LoadUnsafe(ref rDelta, (nuint)i);
                (vTarget + vDelta).StoreUnsafe(ref rTarget, (nuint)i);
            }
        }

        // Stage 4: Scalar tail cleanup (0 to 3 floats)
        for (; i < n; i++)
        {
            target[i] += delta[i];
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
            if (t_tapeStack is { Count: > 0 } stack)
            {
                if (ReferenceEquals(stack.Peek(), this))
                {
                    stack.Pop();
                }
                else if (stack.Contains(this))
                {
                    var remaining = new AutogradTape[stack.Count];
                    int count = 0;
                    while (stack.Count > 0)
                    {
                        var item = stack.Pop();
                        if (!ReferenceEquals(item, this))
                        {
                            remaining[count++] = item;
                        }
                    }
                    for (int i = count - 1; i >= 0; i--)
                    {
                        stack.Push(remaining[i]);
                    }
                }
            }
        }
    }
}
