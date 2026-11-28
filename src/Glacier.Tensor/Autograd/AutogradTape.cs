using System;
using System.Buffers;
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
    private readonly AutogradScratchArena _scratchArena = new();
    private readonly HashSet<Tensor<float>> _intermediateOutputs = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<Tensor<float>, Tensor<float>> _passGrads = new(ReferenceEqualityComparer.Instance);
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
        ArgumentNullException.ThrowIfNull(output);

        // Support legacy callers passing loss inputs (`tape.Backward(predictions)`) by seeding
        // the downstream unseeded loss tensor on the tape instead of writing eager gradients on forward.
        Tensor<float> seedTarget = output;
        for (int idx = _entries.Count - 1; idx >= 0; idx--)
        {
            var e = _entries[idx];
            if (IsLossOp(e.Op) && ReferenceEquals(e.Input0, output) && e.Output.Grad == null)
            {
                seedTarget = e.Output;
                break;
            }
        }

        // Seed output gradient with 1.0f if not already seeded
        bool seededDefault = false;
        if (seedTarget.Grad == null)
        {
            seedTarget.Grad = new Tensor<float>(seedTarget.Shape);
            seedTarget.Grad.Fill(1.0f);
            seededDefault = true;
        }

        bool prevBackward = IsExecutingBackward;
        IsExecutingBackward = true;
        if (!prevBackward)
        {
            _scratchArena.BeginPass();
        }

        HashSet<Tensor<float>> intermediateOutputs = !prevBackward
            ? _intermediateOutputs
            : new HashSet<Tensor<float>>(ReferenceEqualityComparer.Instance);
        Dictionary<Tensor<float>, Tensor<float>> passGrads = !prevBackward
            ? _passGrads
            : new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer.Instance);

        intermediateOutputs.Clear();
        passGrads.Clear();

        try
        {
            Tensor<float> RentTemp(in Shape8 shape) =>
                !prevBackward ? _scratchArena.RentZeroed(shape) : new Tensor<float>(shape);

            for (int i = 0; i < _entries.Count; i++)
            {
                intermediateOutputs.Add(_entries[i].Output);
            }

            var seedPassGrad = RentTemp(seedTarget.Shape);
            if (seededDefault)
            {
                seedPassGrad.Fill(1.0f);
            }
            else if (seedTarget.ElementCount > 0)
            {
                Tensor<float>.CopyElements(seedTarget.Grad, seedPassGrad);
            }
            passGrads[seedTarget] = seedPassGrad;

            void AccumulateTarget(Tensor<float> target, Tensor<float> gradDelta)
            {
                if (intermediateOutputs.Contains(target))
                {
                    if (target.ElementCount == 0 || gradDelta.ElementCount == 0)
                    {
                        return;
                    }

                    if (!passGrads.TryGetValue(target, out var passGrad))
                    {
                        passGrad = RentTemp(target.Shape);
                        passGrads[target] = passGrad;
                    }

                    AccumulateIntoExistingGrad(target.Shape, passGrad, gradDelta);

                    if (!ReferenceEquals(target, seedTarget) && target.Grad != null)
                    {
                        AccumulateIntoExistingGrad(target.Shape, target.Grad, gradDelta);
                    }
                }
                else
                {
                    AccumulateGradient(target, gradDelta);
                }
            }

            // Reverse topological traversal over tape entries
            for (int i = _entries.Count - 1; i >= 0; i--)
            {
                var entry = _entries[i];
                if (!passGrads.TryGetValue(entry.Output, out var dY)) continue;

                switch (entry.Op)
                {
                    case AutogradOp.Add:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                            AccumulateTarget(entry.Input0, dY);
                        if (entry.Input1?.RequiresGrad == true)
                            AccumulateTarget(entry.Input1, dY);
                        break;
                    }

                    case AutogradOp.Subtract:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                            AccumulateTarget(entry.Input0, dY);
                        if (entry.Input1?.RequiresGrad == true)
                        {
                            using var negDy = RentTemp(dY.Shape);
                            ElementwiseKernels.Scale(dY, -1.0f, negDy);
                            AccumulateTarget(entry.Input1, negDy);
                        }
                        break;
                    }

                    case AutogradOp.Multiply:
                    {
                        if (entry.Input0?.RequiresGrad == true && entry.Input1 != null)
                        {
                            using var da = RentTemp(entry.Input0.Shape);
                            ElementwiseKernels.Multiply(dY, entry.Input1, da);
                            AccumulateTarget(entry.Input0, da);
                        }
                        if (entry.Input1?.RequiresGrad == true && entry.Input0 != null)
                        {
                            using var db = RentTemp(entry.Input1.Shape);
                            ElementwiseKernels.Multiply(dY, entry.Input0, db);
                            AccumulateTarget(entry.Input1, db);
                        }
                        break;
                    }

                    case AutogradOp.Scale:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var da = RentTemp(entry.Input0.Shape);
                            ElementwiseKernels.Scale(dY, entry.Scalar, da);
                            AccumulateTarget(entry.Input0, da);
                        }
                        break;
                    }

                    case AutogradOp.MatMul:
                    {
                        var a = entry.Input0!;
                        var b = entry.Input1!;
                        int k = a.Shape[a.Rank - 1];
                        int n = b.Shape[1];
                        int mFlat = k > 0 ? checked((int)(a.ElementCount / k)) : 0;

                        if (mFlat > 0 && k > 0 && n > 0)
                        {
                            using var dYContig = (dY.Rank > 2 && !dY.IsContiguous) ? dY.Contiguous() : null;
                            var dYSrc = dYContig ?? dY;
                            using var dY2d = dY.Rank == 2
                                ? null
                                : new Tensor<float>(dYSrc.MemoryBlock, new Shape8([mFlat, n]), Strides8.ComputeContiguous(new Shape8([mFlat, n])), dYSrc.ElementOffset, addRef: true);
                            var dYMat = dY2d ?? dY;

                            // dA = dY * B^T  ([M, N] * [N, K] -> [M, K])
                            if (a.RequiresGrad)
                            {
                                using var da = RentTemp(a.Shape);
                                using var da2d = a.Rank == 2
                                    ? null
                                    : new Tensor<float>(da.MemoryBlock, new Shape8([mFlat, k]), Strides8.ComputeContiguous(new Shape8([mFlat, k])), da.ElementOffset, addRef: true);
                                GpuAccelerator.AcceleratedMatMul(dYMat, b, da2d ?? da, transposeA: false, transposeB: true, GpuTarget.Auto);
                                AccumulateTarget(a, da);
                            }

                            // dB = A^T * dY  ([K, M] * [M, N] -> [K, N])
                            if (b.RequiresGrad)
                            {
                                using var aContig = (a.Rank > 2 && !a.IsContiguous) ? a.Contiguous() : null;
                                var aSrc = aContig ?? a;
                                using var a2d = a.Rank == 2
                                    ? null
                                    : new Tensor<float>(aSrc.MemoryBlock, new Shape8([mFlat, k]), Strides8.ComputeContiguous(new Shape8([mFlat, k])), aSrc.ElementOffset, addRef: true);
                                using var db = RentTemp(b.Shape);
                                GpuAccelerator.AcceleratedMatMul(a2d ?? a, dYMat, db, transposeA: true, transposeB: false, GpuTarget.Auto);
                                AccumulateTarget(b, db);
                            }
                        }
                        break;
                    }

                    case AutogradOp.ReLU:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = RentTemp(entry.Input0.Shape);
                            ElementwiseKernels.ReLUBackward(dY, entry.Input0, dx);
                            AccumulateTarget(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.Sigmoid:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            // dx = dy * y * (1 - y)
                            var y = entry.Output;
                            using var oneMinusY = RentTemp(y.Shape);
                            oneMinusY.Fill(1.0f);
                            ElementwiseKernels.Subtract(oneMinusY, y, oneMinusY);

                            using var dyTimesY = RentTemp(y.Shape);
                            ElementwiseKernels.Multiply(dY, y, dyTimesY);

                            using var dx = RentTemp(y.Shape);
                            ElementwiseKernels.Multiply(dyTimesY, oneMinusY, dx);
                            AccumulateTarget(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.GELU:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = RentTemp(entry.Input0.Shape);
                            ElementwiseKernels.GELUBackward(dY, entry.Input0, dx);
                            AccumulateTarget(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.Softmax:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = RentTemp(entry.Input0.Shape);
                            ElementwiseKernels.SoftmaxBackward(dY, entry.Output, dx, entry.IntParam0);
                            AccumulateTarget(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.RMSNorm:
                    {
                        var a = entry.Input0;
                        var w = entry.Input1;
                        float eps = entry.Scalar > 0 ? entry.Scalar : 1e-5f;

                        Tensor<float>? da = a?.RequiresGrad == true ? RentTemp(a.Shape) : null;
                        Tensor<float>? dw = w?.RequiresGrad == true ? RentTemp(w.Shape) : null;

                        if (da != null || dw != null)
                        {
                            using var scopedDa = da;
                            using var scopedDw = dw;
                            using var dummyDa = da == null ? RentTemp(a!.Shape) : null;

                            ElementwiseKernels.RMSNormBackward(dY, a!, w, da ?? dummyDa!, dw, eps);

                            if (da != null)
                            {
                                AccumulateTarget(a!, da);
                            }
                            if (dw != null)
                            {
                                AccumulateTarget(w!, dw);
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

                        Tensor<float>? da = a?.RequiresGrad == true ? RentTemp(a.Shape) : null;
                        Tensor<float>? dGamma = gamma?.RequiresGrad == true ? RentTemp(gamma.Shape) : null;
                        Tensor<float>? dBeta = beta?.RequiresGrad == true ? RentTemp(beta.Shape) : null;

                        if (da != null || dGamma != null || dBeta != null)
                        {
                            using var scopedDa = da;
                            using var scopedDGamma = dGamma;
                            using var scopedDBeta = dBeta;
                            using var dummyDa = da == null ? RentTemp(a!.Shape) : null;

                            ElementwiseKernels.LayerNormBackward(dY, a!, gamma, da ?? dummyDa!, dGamma, dBeta, eps);

                            if (da != null) AccumulateTarget(a!, da);
                            if (dGamma != null) AccumulateTarget(gamma!, dGamma);
                            if (dBeta != null) AccumulateTarget(beta!, dBeta);
                        }
                        break;
                    }

                    case AutogradOp.Transpose:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = dY.Transpose(entry.IntParam0, entry.IntParam1);
                            AccumulateTarget(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.Reshape:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            using var dx = dY.Reshape(entry.Input0.Shape.AsSpan());
                            AccumulateTarget(entry.Input0, dx);
                        }
                        break;
                    }

                    case AutogradOp.Slice:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            var src = entry.Input0;
                            if (intermediateOutputs.Contains(src))
                            {
                                if (!passGrads.TryGetValue(src, out var passGrad))
                                {
                                    passGrad = RentTemp(src.Shape);
                                    passGrads[src] = passGrad;
                                }
                                using (var passSlice = passGrad.Slice(entry.IntParam0, entry.IntParam1, entry.IntParam2))
                                {
                                    AccumulateIntoTensor(passSlice, dY);
                                }
                                if (!ReferenceEquals(src, seedTarget) && src.Grad != null)
                                {
                                    using var gradSlice = src.Grad.Slice(entry.IntParam0, entry.IntParam1, entry.IntParam2);
                                    AccumulateIntoTensor(gradSlice, dY);
                                }
                            }
                            else
                            {
                                src.Grad ??= new Tensor<float>(src.Shape);
                                using var gradSlice = src.Grad.Slice(entry.IntParam0, entry.IntParam1, entry.IntParam2);
                                AccumulateIntoTensor(gradSlice, dY);
                            }
                        }
                        break;
                    }

                    case AutogradOp.Contiguous:
                    {
                        if (entry.Input0?.RequiresGrad == true)
                        {
                            AccumulateTarget(entry.Input0, dY);
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
                            using var dPred = RentTemp(pred.Shape);
                            Losses.LossFunctions.ComputeMSEGradient(pred, targets, dPred, dLoss);
                            AccumulateTarget(pred, dPred);
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
                            using var dPred = RentTemp(pred.Shape);
                            Losses.LossFunctions.ComputeBinaryCrossEntropyGradient(pred, targets, dPred, dLoss);
                            AccumulateTarget(pred, dPred);
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
                            using var dLogits = RentTemp(logits.Shape);
                            Losses.LossFunctions.ComputeCrossEntropyGradient(logits, targets, dLogits, dLoss);
                            AccumulateTarget(logits, dLogits);
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
                            using var dLogits = RentTemp(logits.Shape);
                            Losses.LossFunctions.ComputeCategoricalCrossEntropyGradient(logits, targets, dLogits, entry.Scalar, dLoss);
                            AccumulateTarget(logits, dLogits);
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
                            using var dLogits = RentTemp(logits.Shape);
                            Losses.LossFunctions.ComputeBrierScoreGradient(logits, targets, dLogits, dLoss);
                            AccumulateTarget(logits, dLogits);
                        }
                        break;
                    }
                }
            }
        }
        finally
        {
            foreach (var pg in passGrads.Values)
            {
                pg.Dispose();
            }
            passGrads.Clear();
            intermediateOutputs.Clear();
            IsExecutingBackward = prevBackward;
        }

        if (!retainGraph)
        {
            _entries.Clear();
        }
    }

    private static bool IsLossOp(AutogradOp op) =>
        op is AutogradOp.MSELoss
            or AutogradOp.BinaryCrossEntropyLoss
            or AutogradOp.CrossEntropyLoss
            or AutogradOp.CategoricalCrossEntropyLoss
            or AutogradOp.BrierScoreLoss;

    private static float GetScalar(Tensor<float> scalarTensor)
    {
        if (scalarTensor.IsContiguous)
        {
            return scalarTensor.AsSpan()[0];
        }
        Span<int> zeros = stackalloc int[scalarTensor.Rank];
        return scalarTensor[zeros];
    }

    private static unsafe void AccumulateGradient(Tensor<float> target, Tensor<float> gradDelta)
    {
        if (target.Grad == null)
        {
            target.Grad = new Tensor<float>(target.Shape);
        }

        if (target.ElementCount == 0 || gradDelta.ElementCount == 0)
        {
            return;
        }

        AccumulateIntoExistingGrad(target.Shape, target.Grad, gradDelta);
    }

    private static unsafe void AccumulateIntoExistingGrad(in Shape8 targetShape, Tensor<float> targetGrad, Tensor<float> gradDelta)
    {
        if (targetShape.ElementCount == 0 || gradDelta.ElementCount == 0)
        {
            return;
        }

        if (targetShape.Equals(gradDelta.Shape))
        {
            if (!targetGrad.IsContiguous)
            {
                AccumulateIntoTensor(targetGrad, gradDelta);
                return;
            }

            using var deltaContig = gradDelta.IsContiguous ? null : gradDelta.Contiguous();
            AddInPlaceVectorized(targetGrad.AsSpan(), (deltaContig ?? gradDelta).AsSpan());
            return;
        }

        // Right-aligned NumPy/PyTorch broadcast reduction
        using var gradDeltaContig = gradDelta.IsContiguous ? null : gradDelta.Contiguous();
        var deltaSrc = gradDeltaContig ?? gradDelta;
        var spanDelta = deltaSrc.AsSpan();

        int deltaRank = gradDelta.Rank;
        int targetRank = targetShape.Rank;
        if (targetRank > deltaRank)
        {
            throw new InvalidOperationException($"Cannot broadcast-reduce gradient from {gradDelta.Shape} to higher-rank shape {targetShape}.");
        }

        int offsetRank = deltaRank - targetRank;
        for (int k = 0; k < deltaRank; k++)
        {
            int tIdx = k - offsetRank;
            int tDim = tIdx >= 0 ? targetShape[tIdx] : 1;
            int dDim = gradDelta.Shape[k];
            if (tDim != dDim && tDim != 1)
            {
                throw new InvalidOperationException($"Incompatible broadcast shapes in gradient accumulation: {gradDelta.Shape} -> {targetShape}.");
            }
        }

        // Fast path: leading-only reduction (all target dims >= 1 trailing match delta trailing dims, only leading dims are 1)
        bool leadingOnly = true;
        bool seenNonOne = false;
        for (int k = 0; k < targetRank; k++)
        {
            int tDim = targetShape[k];
            int dDim = gradDelta.Shape[offsetRank + k];
            if (seenNonOne && tDim != dDim)
            {
                leadingOnly = false;
                break;
            }
            if (tDim > 1)
            {
                seenNonOne = true;
            }
        }

        if (targetGrad.IsContiguous && leadingOnly)
        {
            var spanTarget = targetGrad.AsSpan();
            int d = spanTarget.Length;
            int batch = spanDelta.Length / d;
            for (int b = 0; b < batch; b++)
            {
                AddInPlaceVectorized(spanTarget, spanDelta.Slice(b * d, d));
            }
            return;
        }

        // General right-aligned stride-0 reduction (e.g. [M, 1] <- [M, N], [B, 1, D] <- [B, S, D], [1] <- [...])
        using var targetContig = targetGrad.IsContiguous ? null : targetGrad.Contiguous();
        var targetDst = targetContig ?? targetGrad;
        var dstSpan = targetDst.AsSpan();

        Span<long> tStrides = stackalloc long[deltaRank];
        for (int k = 0; k < deltaRank; k++)
        {
            int tIdx = k - offsetRank;
            if (tIdx < 0 || targetShape[tIdx] == 1)
            {
                tStrides[k] = 0;
            }
            else
            {
                tStrides[k] = targetDst.LongStrides[tIdx];
            }
        }

        Span<int> coords = stackalloc int[deltaRank];
        long dstOffset = 0;
        for (int i = 0; i < spanDelta.Length; i++)
        {
            dstSpan[(int)dstOffset] += spanDelta[i];

            for (int d = deltaRank - 1; d >= 0; d--)
            {
                coords[d]++;
                if (coords[d] < gradDelta.Shape[d])
                {
                    dstOffset += tStrides[d];
                    break;
                }

                dstOffset -= (long)(coords[d] - 1) * tStrides[d];
                coords[d] = 0;
            }
        }

        if (targetContig != null)
        {
            Tensor<float>.CopyElements(targetContig, targetGrad);
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
            _scratchArena.Dispose();
            if (t_tapeStack is { Count: > 0 } stack)
            {
                if (ReferenceEquals(stack.Peek(), this))
                {
                    stack.Pop();
                }
                else if (stack.Contains(this))
                {
                    var remaining = ArrayPool<AutogradTape>.Shared.Rent(stack.Count);
                    int count = 0;
                    try
                    {
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
                    finally
                    {
                        ArrayPool<AutogradTape>.Shared.Return(remaining, clearArray: true);
                    }
                }
            }
        }
    }
}
