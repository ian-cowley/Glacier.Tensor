using System;
using System.Collections.Generic;
using Glacier.Compute.Assembler.SpirV;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tensor.Interop;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;
using Xunit;

namespace Glacier.Tensor.Tests;

public unsafe class M31HardeningTests
{
    private sealed class FlagLifetimeOwner : IDisposable
    {
        public bool Disposed;
        public void Dispose() => Disposed = true;
    }

    // -------------------------------------------------------------------------
    // 1. Core Memory Safety, Shape8, Strides8 & Polaris ValidityMask
    // -------------------------------------------------------------------------

    [Fact]
    public void Shape8_RankZeroScalar_HasElementCountOne_AndTensorScalarWorks()
    {
        var scalarShape = default(Shape8);
        Assert.Equal(0, scalarShape.Rank);
        Assert.Equal(1L, scalarShape.ElementCount);

        using var scalar = new Tensor<float>(scalarShape);
        Assert.Equal(0, scalar.Rank);
        Assert.Equal(1L, scalar.ElementCount);
        scalar.Fill(7.5f);
        Assert.Equal(7.5f, scalar.AsSpan()[0]);
    }

    [Fact]
    public void Strides8_SupportsLarge64BitStrides_SwapAndContiguity()
    {
        var shape = Shape8.Create(4, 1024, 1024);
        var strides = Strides8.ComputeContiguous(shape);
        Assert.Equal(1024L * 1024L, strides[0]);
        Assert.Equal(1024L, strides[1]);
        Assert.Equal(1L, strides[2]);
        Assert.True(shape.IsContiguous(in strides));

        var swapped = strides.SwapDimensions(0, 2);
        Assert.Equal(1L, swapped[0]);
        Assert.Equal(1024L, swapped[1]);
        Assert.Equal(1024L * 1024L, swapped[2]);
        Assert.False(shape.IsContiguous(in swapped));

        var bigStrides = Strides8.Create(5_000_000_000L, 1L);
        Assert.Equal(5_000_000_000L, bigStrides[0]);
        Assert.Throws<OverflowException>(() => bigStrides.ToShape8());
    }

    [Fact]
    public void NativeMemoryBlock_ExternalNullPointer_AndByteOverflow_ThrowSafely()
    {
        Assert.Throws<ArgumentNullException>(() => new NativeMemoryBlock<float>(null, 10, ownsMemory: false));
        Assert.Throws<OverflowException>(() => new NativeMemoryBlock<double>(long.MaxValue / 2));
    }

    [Fact]
    public void Tensor_SharedMemoryCtor_AddRefsBlock_SoDisposingLocalBlockKeepsMemoryValid()
    {
        var sharedBlock = new NativeMemoryBlock<float>(16);
        sharedBlock.AsSpan()[5] = 42.25f;

        var shape = Shape8.Create(4, 4);
        var view = new Tensor<float>(sharedBlock, shape, Shape8.ComputeContiguousStrides(shape), 0);

        // Caller disposes its initial reference; Tensor's AddRef must keep sharedBlock alive
        sharedBlock.Dispose();
        Assert.False(sharedBlock.IsDisposed);
        Assert.Equal(42.25f, view[1, 1]);

        view.Dispose();
        Assert.True(sharedBlock.IsDisposed);
    }

    [Fact]
    public void NativeMemoryBlock_Finalizer_DisposesLifetimeOwner()
    {
        var flagOwner = new FlagLifetimeOwner();
        void AllocateUnDisposed()
        {
            var handle = new float[4].AsMemory().Pin();
            var block = new NativeMemoryBlock<float>((float*)handle.Pointer, 4, ownsMemory: false, lifetimeOwner: new CompositeOwner(handle, flagOwner));
            GC.KeepAlive(block);
        }

        AllocateUnDisposed();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Assert.True(flagOwner.Disposed);
    }

    private sealed class CompositeOwner(System.Buffers.MemoryHandle handle, FlagLifetimeOwner flag) : IDisposable
    {
        public void Dispose()
        {
            handle.Dispose();
            flag.Dispose();
        }
    }

    [Fact]
    public void Polaris_AsTensorView_And_ToTensor_GuardNullValidityMask_AndFillOverloadWorks()
    {
        var col = new Float32Series("vals", 4);
        col.Memory.Span[0] = 1.0f;
        col.Memory.Span[1] = 99.0f;
        col.Memory.Span[2] = 3.0f;
        col.Memory.Span[3] = 4.0f;
        col.ValidityMask.SetNull(1);

        Assert.Throws<InvalidOperationException>(() => col.AsTensorView());

        var df = new DataFrame([col]);
        Assert.Throws<InvalidOperationException>(() => df.ToTensor("vals"));

        using var filled = df.ToTensor(nullFillValue: -5.0f, "vals");
        Assert.Equal(1.0f, filled[0, 0]);
        Assert.Equal(-5.0f, filled[1, 0]);
        Assert.Equal(3.0f, filled[2, 0]);
        Assert.Equal(4.0f, filled[3, 0]);
    }

    // -------------------------------------------------------------------------
    // 2. SPIR-V Bytecode Validation
    // -------------------------------------------------------------------------

    [Fact]
    public void VulkanGemmKernel_EmbeddedSpirvBytecode_PassesSpirvValidator()
    {
        bool valid = SpirvValidator.Validate(VulkanGemmKernel.SpvBytecode, out string? error);
        Assert.True(valid, $"SpirvValidator failed: {error}");
    }

    // -------------------------------------------------------------------------
    // 3. SimdMath (Opt #16), GemmKernels, ElementwiseKernels & Optimizers
    // -------------------------------------------------------------------------

    [Fact]
    public void SimdMath_FastExpAndTanhAndWidenSum_MatchMathFReference()
    {
        float[] testVals = [-95f, -50f, -10f, -2.5f, -0.1f, 0f, 0.25f, 3.5f, 12f, 95f];
        foreach (float x in testVals)
        {
            if (x <= 0f)
            {
                float np = SimdMath.FastExpNonPositiveScalar(x);
                float expectedNp = MathF.Exp(x);
                float tol = MathF.Max(1e-6f, expectedNp * 1e-4f);
                Assert.True(MathF.Abs(np - expectedNp) <= tol, $"FastExpNonPositiveScalar({x}): got {np}, exp {expectedNp}");
            }

            float e = SimdMath.FastExpScalar(x);
            float expRef = MathF.Exp(x);
            if (float.IsInfinity(expRef))
            {
                Assert.True(float.IsPositiveInfinity(e));
            }
            else
            {
                float tol = MathF.Max(1e-6f, expRef * 2e-4f);
                Assert.True(MathF.Abs(e - expRef) <= tol, $"FastExpScalar({x}): got {e}, exp {expRef}");
            }

            float t = SimdMath.FastTanhScalar(x);
            Assert.True(MathF.Abs(t - MathF.Tanh(x)) <= 1e-5f, $"FastTanhScalar({x}): got {t}, exp {MathF.Tanh(x)}");
        }

        float[] arr = new float[257];
        double refSum = 0.0;
        double refSq = 0.0;
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = 1000.0f + (i % 7) * 0.125f;
            refSum += arr[i];
            refSq += (double)arr[i] * arr[i];
        }

        fixed (float* p = arr)
        {
            Assert.Equal(refSum, SimdMath.WidenSum(p, arr.Length), 6);
            Assert.Equal(refSq, SimdMath.WidenSumSquares(p, arr.Length), 3);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void GemmKernels_AllTransposeModes_AndAdvSimd4x4_MatchNaiveReference(bool transposeA, bool transposeB)
    {
        int m = 69, k = 67, n = 73;
        using var a = TensorFloatExtensions.RandomUniform(transposeA ? [k, m] : [m, k], -1f, 1f, seed: 11);
        using var b = TensorFloatExtensions.RandomUniform(transposeB ? [n, k] : [k, n], -1f, 1f, seed: 22);
        using var c = new Tensor<float>(m, n);

        GemmKernels.MatMul(a, b, c, transposeA, transposeB);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < k; p++)
                {
                    float va = transposeA ? a[p, i] : a[i, p];
                    float vb = transposeB ? b[j, p] : b[p, j];
                    sum += (double)va * vb;
                }
                Assert.True(MathF.Abs(c[i, j] - (float)sum) <= 1e-4f,
                    $"MatMul({transposeA},{transposeB}) mismatch at [{i},{j}]: {c[i, j]} vs {sum}");
            }
        }

        // Verify MicroKernelGemmAdvSimd_4x4 directly as well
        float[] a4 = new float[4 * 9];
        float[] b4 = new float[9 * 4];
        float[] c4 = new float[4 * 4];
        for (int i = 0; i < a4.Length; i++) a4[i] = (i % 5) * 0.25f;
        for (int i = 0; i < b4.Length; i++) b4[i] = ((i + 2) % 7) * 0.2f;
        fixed (float* pa = a4, pb = b4, pc = c4)
        {
            GemmKernels.MicroKernelGemmAdvSimd_4x4(pa, pb, pc, 9, 4, 4, 9);
        }
        for (int r = 0; r < 4; r++)
        {
            for (int col = 0; col < 4; col++)
            {
                float exp = 0f;
                for (int p = 0; p < 9; p++) exp += a4[r * 9 + p] * b4[p * 4 + col];
                Assert.Equal(exp, c4[r * 4 + col], 4);
            }
        }
    }

    [Fact]
    public void ElementwiseKernels_NonContiguousOutput_And_ExtremeSigmoid_WorkAccurately()
    {
        // Extreme Sigmoid must never produce NaN on -100..+100
        using var xExt = Tensor<float>.FromArray([-100f, -20f, 0f, 20f, 100f], 5);
        using var yExt = new Tensor<float>(5);
        ElementwiseKernels.Sigmoid(xExt, yExt);
        for (int i = 0; i < 5; i++)
        {
            Assert.False(float.IsNaN(yExt[i]));
            float exp = 1.0f / (1.0f + MathF.Exp(-xExt[i]));
            Assert.True(MathF.Abs(yExt[i] - exp) <= 1e-4f);
        }

        // Non-contiguous transposed output in ElementwiseKernels.Add & ReLU
        using var a = Tensor<float>.FromArray([1f, -2f, 3f, -4f, 5f, -6f], 2, 3);
        using var b = Tensor<float>.FromArray([10f, 20f, 30f, 40f, 50f, 60f], 2, 3);
        using var outBacking = new Tensor<float>(3, 2);
        using var outTransposed = outBacking.Transpose(); // shape [2, 3], non-contiguous

        ElementwiseKernels.Add(a, b, outTransposed);
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                Assert.Equal(a[r, c] + b[r, c], outTransposed[r, c], 5);
                Assert.Equal(a[r, c] + b[r, c], outBacking[c, r], 5);
            }
        }
    }

    [Fact]
    public void AdamW_And_SGD_SimdUpdates_MatchScalarFormula()
    {
        const int n = 37; // exercises AVX2/Vec128 + scalar tail
        using var wAdam = TensorFloatExtensions.RandomUniform([n], -1f, 1f, seed: 1);
        using var wSgd = TensorFloatExtensions.RandomUniform([n], -1f, 1f, seed: 1);
        wAdam.Grad = TensorFloatExtensions.RandomUniform([n], -0.5f, 0.5f, seed: 2);
        wSgd.Grad = TensorFloatExtensions.RandomUniform([n], -0.5f, 0.5f, seed: 2);

        float[] wOrig = wAdam.AsSpan().ToArray();
        float[] gOrig = wAdam.Grad.AsSpan().ToArray();

        using var adam = new AdamW([wAdam], lr: 0.01f, beta1: 0.9f, beta2: 0.999f, eps: 1e-8f, weightDecay: 0.05f);
        adam.Step();

        float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.05f;
        float invBc1 = 1.0f / (1.0f - b1);
        float invBc2 = 1.0f / (1.0f - b2);
        for (int i = 0; i < n; i++)
        {
            float m = (1f - b1) * gOrig[i];
            float v = (1f - b2) * gOrig[i] * gOrig[i];
            float exp = wOrig[i] * (1f - lr * wd) - lr * (m * invBc1) / (MathF.Sqrt(v * invBc2) + eps);
            Assert.True(MathF.Abs(wAdam[i] - exp) <= 1e-5f);
        }

        using var sgd = new SGD([wSgd], lr: 0.02f, momentum: 0.9f, weightDecay: 0.01f);
        sgd.Step();
        for (int i = 0; i < n; i++)
        {
            float gradWd = gOrig[i] + 0.01f * wOrig[i];
            float exp = wOrig[i] - 0.02f * gradWd;
            Assert.True(MathF.Abs(wSgd[i] - exp) <= 1e-5f);
        }
    }

    // -------------------------------------------------------------------------
    // 4. Opt #14 Host Primitives (AutogradScratchWorkspace, BaseBackward, FusedCE)
    // -------------------------------------------------------------------------

    [Fact]
    public void AutogradScratchWorkspace_GetOrCreate_Wrap_AndDisposeAllThreadLocal()
    {
        var ws = AutogradScratchWorkspace.GetOrCreate(seqLen: 8, hiddenDim: 16, ffnDim: 32, nHeadsQ: 2, nHeadsKv: 1, headDim: 8);
        Assert.False(ws.IsDisposed);
        using (var wrapped = AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, 8, 16))
        {
            wrapped[2, 3] = 9.0f;
            Assert.Equal(9.0f, ws.HiddenScratchA[2 * 16 + 3]);
        }

        // Disposing wrapped view must not free workspace slab
        Assert.False(ws.IsDisposed);
        AutogradScratchWorkspace.DisposeAllThreadLocal();
        Assert.True(ws.IsDisposed);
    }

    [Fact]
    public void TensorBackwardKernels_And_FusedCrossEntropyLoss_MultiTile_MatchNaiveFp64()
    {
        int seqLen = 3;
        int hiddenDim = 17;
        int vocabSize = 4150; // > VocabTileSize (4096) to exercise multi-tile streaming path

        var rng = new Random(12345);
        float[] hidden = new float[seqLen * hiddenDim];
        float[] weight = new float[vocabSize * hiddenDim];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        int[] targets = [5, 4120, -100]; // last token masked out

        float[] dHidden = new float[seqLen * hiddenDim];
        float fusedLossFwdOnly;
        float fusedLoss;
        fixed (float* pH = hidden, pW = weight, pDH = dHidden)
        {
            fusedLossFwdOnly = FusedCrossEntropyLoss.ComputeLoss(pH, targets, pW, seqLen, hiddenDim, vocabSize, out int validCount);
            Assert.Equal(2, validCount);
            fusedLoss = FusedCrossEntropyLoss.ComputeLossAndGradients(pH, targets, pW, pDH, seqLen, hiddenDim, vocabSize);
        }

        Assert.True(MathF.Abs(fusedLoss - fusedLossFwdOnly) <= 1e-4f);

        // Naive FP64 reference
        double refLoss = 0.0;
        double[] refGrad = new double[seqLen * hiddenDim];
        int valid = 2;
        for (int s = 0; s < valid; s++)
        {
            double[] logits = new double[vocabSize];
            double maxL = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double dot = 0.0;
                for (int d = 0; d < hiddenDim; d++) dot += (double)hidden[s * hiddenDim + d] * weight[v * hiddenDim + d];
                logits[v] = dot;
                if (dot > maxL) maxL = dot;
            }
            double sumExp = 0.0;
            for (int v = 0; v < vocabSize; v++) sumExp += Math.Exp(logits[v] - maxL);
            refLoss += (maxL + Math.Log(sumExp)) - logits[targets[s]];

            for (int v = 0; v < vocabSize; v++)
            {
                double p = Math.Exp(logits[v] - maxL) / sumExp;
                double dy = (p - (v == targets[s] ? 1.0 : 0.0)) / valid;
                for (int d = 0; d < hiddenDim; d++) refGrad[s * hiddenDim + d] += dy * weight[v * hiddenDim + d];
            }
        }
        refLoss /= valid;

        Assert.True(Math.Abs(fusedLoss - refLoss) <= 1e-4, $"FusedCE loss {fusedLoss} vs ref {refLoss}");
        for (int i = 0; i < hiddenDim * 2; i++)
        {
            Assert.True(Math.Abs(dHidden[i] - refGrad[i]) <= 5e-4, $"dHidden[{i}]: {dHidden[i]} vs {refGrad[i]}");
        }
        // Masked token 2 gradient must remain zero
        for (int d = 0; d < hiddenDim; d++)
        {
            Assert.Equal(0.0f, dHidden[2 * hiddenDim + d]);
        }
    }

    // -------------------------------------------------------------------------
    // 5. AutogradTape Diamond/Multi-Branch, Broadcast Reduction & 3D Layers/Losses
    // -------------------------------------------------------------------------

    [Fact]
    public void AutogradTape_MultiBranchLoss_AccumulatesWithoutClobbering()
    {
        using var x = Tensor<float>.FromArray([1.0f, 2.0f, -1.0f, 0.5f], 2, 2);
        using var t1 = Tensor<float>.FromArray([0.0f, 0.0f, 0.0f, 0.0f], 2, 2);
        using var t2 = Tensor<float>.FromArray([2.0f, 2.0f, 2.0f, 2.0f], 2, 2);

        using var tape = new AutogradTape();
        tape.Watch(x);

        using var y = TensorOps.Scale(x, 2.0f);
        var (_, l1) = LossFunctions.MSELoss(y, t1);
        var (_, l2) = LossFunctions.MSELoss(y, t2);
        using var l1Ref = l1;
        using var l2Ref = l2;
        using var total = TensorOps.Add(l1, l2);

        tape.Backward(total);

        // d/dx [ 0.25*sum((2x)^2) + 0.25*sum((2x - 2)^2) ] = 2x + (2x - 2) = 4x - 2
        Assert.NotNull(x.Grad);
        for (int i = 0; i < 4; i++)
        {
            float expectedGrad = 4.0f * x.AsSpan()[i] - 2.0f;
            Assert.True(MathF.Abs(x.Grad.AsSpan()[i] - expectedGrad) <= 1e-4f,
                $"Diamond loss grad[{i}]: got {x.Grad.AsSpan()[i]}, expected {expectedGrad}");
        }
    }

    [Fact]
    public void AutogradTape_LegacyBackwardOnPredictions_MatchesLossTensorBackward()
    {
        using var w1 = Tensor<float>.FromArray([0.5f, -0.3f, 0.2f, 0.8f], 2, 2);
        using var w2 = Tensor<float>.FromArray([0.5f, -0.3f, 0.2f, 0.8f], 2, 2);
        using var x = Tensor<float>.FromArray([1.0f, -2.0f, 0.5f, 1.5f], 2, 2);
        using var target = Tensor<float>.FromArray([1.0f, 0.0f, 0.0f, 1.0f], 2, 2);

        using (var tape1 = new AutogradTape())
        {
            tape1.Watch(w1);
            using var pred1 = TensorOps.MatMul(x, w1);
            var (_, loss1) = LossFunctions.MSELoss(pred1, target);
            using var _l1 = loss1;
            tape1.Backward(loss1);
        }

        using (var tape2 = new AutogradTape())
        {
            tape2.Watch(w2);
            using var pred2 = TensorOps.MatMul(x, w2);
            var (_, loss2) = LossFunctions.MSELoss(pred2, target);
            using var _l2 = loss2;
            tape2.Backward(pred2); // legacy API pattern
        }

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(w1.Grad!.AsSpan()[i], w2.Grad!.AsSpan()[i], 5);
        }
    }

    [Fact]
    public void AutogradTape_RightAlignedBroadcastReduction_HandlesMiddleAndTrailingAxes()
    {
        // Target [2, 1], gradOutput [2, 3] -> must sum columns per row (NOT modulo stride!)
        using var a = Tensor<float>.FromArray([1.0f, 2.0f], 2, 1);
        using var b = Tensor<float>.Zeros(2, 3);
        using var outSum = new Tensor<float>(2, 3);
        using var tape = new AutogradTape();
        tape.Watch(a);

        // Simulate broadcasted add: outSum[r, c] = a[r, 0] + b[r, c]
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 3; c++)
                outSum[r, c] = a[r, 0] + b[r, c];
        outSum.RequiresGrad = true;
        tape.Record(new TapeEntry(AutogradOp.Add, outSum, a, b));

        outSum.Grad = Tensor<float>.FromArray([1f, 2f, 3f, 10f, 20f, 30f], 2, 3);
        tape.Backward(outSum);

        Assert.Equal(6.0f, a.Grad![0, 0], 5);
        Assert.Equal(60.0f, a.Grad![1, 0], 5);

        // Middle axis broadcast: target [2, 1, 2], gradOutput [2, 3, 2]
        using var m = Tensor<float>.FromArray([1f, 2f, 3f, 4f], 2, 1, 2);
        using var n = Tensor<float>.Zeros(2, 3, 2);
        using var out3d = Tensor<float>.Zeros(2, 3, 2);
        out3d.RequiresGrad = true;
        using var tape2 = new AutogradTape();
        tape2.Watch(m);
        tape2.Record(new TapeEntry(AutogradOp.Add, out3d, m, n));
        out3d.Grad = Tensor<float>.FromArray([1f, 10f, 2f, 20f, 3f, 30f, 4f, 40f, 5f, 50f, 6f, 60f], 2, 3, 2);
        tape2.Backward(out3d);

        Assert.Equal(6.0f, m.Grad![0, 0, 0], 5);
        Assert.Equal(60.0f, m.Grad![0, 0, 1], 5);
        Assert.Equal(15.0f, m.Grad![1, 0, 0], 5);
        Assert.Equal(150.0f, m.Grad![1, 0, 1], 5);
    }

    [Fact]
    public void Linear_LoraLinear_And_Losses_SupportRank3BatchSeqFeatures()
    {
        int b = 2, s = 3, dIn = 4, dOut = 5;
        using var linear = new Linear(dIn, dOut, seed: 42);
        using var baseW = TensorFloatExtensions.RandomUniform([dIn, dOut], -0.5f, 0.5f, seed: 43);
        using var lora = new LoraLinear(baseW, rank: 2, alpha: 4f, seed: 44);
        using var x3d = TensorFloatExtensions.RandomUniform([b, s, dIn], -1f, 1f, seed: 45);
        using var targets = Tensor<float>.FromArray([1f, 0f, 3f, 2f, 4f, 1f], b, s);

        using var tape = new AutogradTape();
        foreach (var p in linear.Parameters) tape.Watch(p);

        using var linOut = linear.Forward(x3d);
        Assert.Equal(3, linOut.Rank);
        Assert.Equal(b, linOut.Shape[0]);
        Assert.Equal(s, linOut.Shape[1]);
        Assert.Equal(dOut, linOut.Shape[2]);

        var (lossVal, lossTensor) = LossFunctions.CrossEntropy(linOut, targets);
        using var _loss = lossTensor;
        Assert.True(lossVal > 0f && float.IsFinite(lossVal));

        tape.Backward(lossTensor);

        bool hasWGrad = false;
        foreach (float g in linear.Weight.Grad!.AsSpan()) if (MathF.Abs(g) > 1e-6f) { hasWGrad = true; break; }
        bool hasBGrad = false;
        foreach (float g in linear.Bias.Grad!.AsSpan()) if (MathF.Abs(g) > 1e-6f) { hasBGrad = true; break; }
        Assert.True(hasWGrad);
        Assert.True(hasBGrad);

        using var loraOut = lora.Forward(x3d);
        Assert.Equal(3, loraOut.Rank);
        Assert.Equal(b, loraOut.Shape[0]);
        Assert.Equal(s, loraOut.Shape[1]);
        Assert.Equal(dOut, loraOut.Shape[2]);
    }

    // -------------------------------------------------------------------------
    // 6. AutogradTape retainGraph: true Multi-Loss Isolation Regressions
    // -------------------------------------------------------------------------

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AutogradTape_RetainGraphMultiLoss_DoesNotDoubleCountIntermediates(bool zeroFirstLossGrad)
    {
        using var rx = Tensor<float>.FromArray([1f, 2f], 1, 2);
        using var rt1 = Tensor<float>.FromArray([0f, 0f], 1, 2);
        using var rt2 = Tensor<float>.FromArray([1f, 1f], 1, 2);

        using var tape = new AutogradTape();
        tape.Watch(rx);

        using var ry = TensorOps.Scale(rx, 3f);
        var (_, l1) = LossFunctions.MSELoss(ry, rt1);
        var (_, l2) = LossFunctions.MSELoss(ry, rt2);
        using var _l1 = l1;
        using var _l2 = l2;

        // Pass 1: d(l1)/d(ry) = (ry - rt1) = [3, 6], d(l1)/d(rx) = 3 * [3, 6] = [9, 18]
        tape.Backward(l1, retainGraph: true);
        Assert.NotNull(l1.Grad);
        Assert.Equal(9f, rx.Grad![0, 0], 4);
        Assert.Equal(18f, rx.Grad![0, 1], 4);
        Assert.Null(ry.Grad);

        if (zeroFirstLossGrad)
        {
            l1.Grad!.Fill(0f);
        }

        // Pass 2: d(l2)/d(ry) = (ry - rt2) = [2, 5], d(l2)/d(rx) = [6, 15]
        // Accumulated rx.Grad must be [9 + 6, 18 + 15] = [15, 33] (never [24, 51]).
        tape.Backward(l2, retainGraph: false);
        Assert.Equal(15f, rx.Grad![0, 0], 4);
        Assert.Equal(33f, rx.Grad![0, 1], 4);
    }

    [Fact]
    public void AutogradTape_RetainGraph_WatchedIntermediate_AccumulatesOwnGradWithoutUpstreamDoubleCount()
    {
        using var rx = Tensor<float>.FromArray([1f, 2f], 1, 2);
        using var rt1 = Tensor<float>.FromArray([0f, 0f], 1, 2);
        using var rt2 = Tensor<float>.FromArray([1f, 1f], 1, 2);

        using var tape = new AutogradTape();
        tape.Watch(rx);

        using var ry = TensorOps.Scale(rx, 3f);
        tape.Watch(ry);

        var (_, l1) = LossFunctions.MSELoss(ry, rt1);
        var (_, l2) = LossFunctions.MSELoss(ry, rt2);
        using var _l1 = l1;
        using var _l2 = l2;

        tape.Backward(l1, retainGraph: true);
        Assert.Equal(3f, ry.Grad![0, 0], 4);
        Assert.Equal(6f, ry.Grad![0, 1], 4);

        tape.Backward(l2, retainGraph: false);
        Assert.Equal(5f, ry.Grad![0, 0], 4);
        Assert.Equal(11f, ry.Grad![0, 1], 4);
        Assert.Equal(15f, rx.Grad![0, 0], 4);
        Assert.Equal(33f, rx.Grad![0, 1], 4);
    }

    [Fact]
    public void AutogradTape_RetainGraph_LinearDiamondAndSlice_MatchFreshSinglePassSum()
    {
        static (float[] dX, float[] dW, float[] dB) RunSinglePass(bool secondLoss)
        {
            using var linear = new Linear(3, 4, seed: 7);
            using var x = Tensor<float>.FromArray([0.5f, -1.0f, 1.5f, 2.0f, 0.25f, -0.5f], 2, 3);
            using var mask = Tensor<float>.FromArray([1.0f, 0.5f, 2.0f, -1.5f], 2, 2);
            using var t1 = Tensor<float>.FromArray([0.1f, -0.2f, 0.3f, 0.4f], 2, 2);
            using var t2 = Tensor<float>.FromArray([1.0f, 0.5f, -0.5f, 0.2f], 2, 2);

            using var tape = new AutogradTape();
            tape.Watch(x);
            foreach (var p in linear.Parameters) tape.Watch(p);

            using var h = linear.Forward(x);
            using var act = TensorOps.ReLU(h);
            using var sl = act.Slice(1, 0, 2);
            using var branchA = TensorOps.Scale(sl, 1.5f);
            using var branchB = TensorOps.Multiply(sl, mask);
            using var diamond = TensorOps.Add(branchA, branchB);

            var (_, l1) = LossFunctions.MSELoss(diamond, t1);
            var (_, l2) = LossFunctions.MSELoss(diamond, t2);
            using var _l1 = l1;
            using var _l2 = l2;

            tape.Backward(secondLoss ? l2 : l1, retainGraph: false);
            return (x.Grad!.ToArray(), linear.Weight.Grad!.ToArray(), linear.Bias.Grad!.ToArray());
        }

        var (dX1, dW1, dB1) = RunSinglePass(secondLoss: false);
        var (dX2, dW2, dB2) = RunSinglePass(secondLoss: true);

        using var linearRet = new Linear(3, 4, seed: 7);
        using var xRet = Tensor<float>.FromArray([0.5f, -1.0f, 1.5f, 2.0f, 0.25f, -0.5f], 2, 3);
        using var maskRet = Tensor<float>.FromArray([1.0f, 0.5f, 2.0f, -1.5f], 2, 2);
        using var t1Ret = Tensor<float>.FromArray([0.1f, -0.2f, 0.3f, 0.4f], 2, 2);
        using var t2Ret = Tensor<float>.FromArray([1.0f, 0.5f, -0.5f, 0.2f], 2, 2);

        using var tapeRet = new AutogradTape();
        tapeRet.Watch(xRet);
        foreach (var p in linearRet.Parameters) tapeRet.Watch(p);

        using var hRet = linearRet.Forward(xRet);
        using var actRet = TensorOps.ReLU(hRet);
        using var slRet = actRet.Slice(1, 0, 2);
        using var branchARet = TensorOps.Scale(slRet, 1.5f);
        using var branchBRet = TensorOps.Multiply(slRet, maskRet);
        using var diamondRet = TensorOps.Add(branchARet, branchBRet);

        var (_, l1Ret) = LossFunctions.MSELoss(diamondRet, t1Ret);
        var (_, l2Ret) = LossFunctions.MSELoss(diamondRet, t2Ret);
        using var _l1Ret = l1Ret;
        using var _l2Ret = l2Ret;

        tapeRet.Backward(l1Ret, retainGraph: true);
        tapeRet.Backward(l2Ret, retainGraph: false);

        var actualDX = xRet.Grad!.ToArray();
        var actualDW = linearRet.Weight.Grad!.ToArray();
        var actualDB = linearRet.Bias.Grad!.ToArray();

        for (int i = 0; i < dX1.Length; i++) Assert.True(MathF.Abs(actualDX[i] - (dX1[i] + dX2[i])) <= 1e-4f);
        for (int i = 0; i < dW1.Length; i++) Assert.True(MathF.Abs(actualDW[i] - (dW1[i] + dW2[i])) <= 1e-4f);
        for (int i = 0; i < dB1.Length; i++) Assert.True(MathF.Abs(actualDB[i] - (dB1[i] + dB2[i])) <= 1e-4f);
    }

    [Fact]
    public void AutogradScratchWorkspace_FailedAllocation_ResetsDimensions_And_RecoversCleanly()
    {
        try
        {
            using var ws = new AutogradScratchWorkspace(4, 16, 32, 2, 2, 8);
            Assert.True(ws.FfnScratchA != null);

            // Trigger an allocation failure (4 TB request -> OutOfMemoryException)
            Assert.ThrowsAny<Exception>(() => ws.EnsureCapacity(1_000_000, 16, 1_000_000, 2, 2, 8));
            Assert.True(ws.FfnScratchA == null);
            Assert.True(ws.KvDimScratchB == null);

            // Subsequent EnsureCapacity with smaller dims must re-allocate instead of skipping on stale _maxSeqLen
            ws.EnsureCapacity(4, 16, 32, 2, 2, 8);
            Assert.True(ws.FfnScratchA != null);
            Assert.True(ws.KvDimScratchB != null);

            // ThreadStatic GetOrCreate must also recover cleanly when EnsureCapacity fails
            var tlsWs = AutogradScratchWorkspace.GetOrCreate(4, 16, 32, 2, 2, 8);
            Assert.True(tlsWs.FfnScratchA != null);
            Assert.ThrowsAny<Exception>(() => AutogradScratchWorkspace.GetOrCreate(1_000_000, 16, 1_000_000, 2, 2, 8));

            var recovered = AutogradScratchWorkspace.GetOrCreate(4, 16, 32, 2, 2, 8);
            Assert.True(recovered.FfnScratchA != null);
            Assert.True(recovered.KvDimScratchB != null);

            // Wrap guards null pointers and non-positive / rank-0 shapes
            Assert.Throws<ArgumentNullException>(() => AutogradScratchWorkspace.Wrap(null, 2, 2));
            Assert.Throws<ArgumentOutOfRangeException>(() => AutogradScratchWorkspace.Wrap(recovered.FfnScratchA));
            Assert.Throws<ArgumentOutOfRangeException>(() => AutogradScratchWorkspace.Wrap(recovered.FfnScratchA, 0, 4));
        }
        finally
        {
            AutogradScratchWorkspace.DisposeAllThreadLocal();
        }
    }
}
