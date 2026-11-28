using System;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;

namespace Glacier.Tensor.Core;

/// <summary>
/// Differentiable high-level tensor operations that seamlessly integrate with AutogradTape.
/// </summary>
public static class TensorOps
{
    public static Tensor<float> MatMul(Tensor<float> a, Tensor<float> b, GpuTarget target = GpuTarget.Auto)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Rank < 2 || b.Rank != 2)
            throw new ArgumentException("MatMul expects a Rank >= 2 first operand and a 2D second operand.");

        int k = a.Shape[a.Rank - 1];
        if (k != b.Shape[0])
            throw new ArgumentException($"Inner dimension mismatch: {k} != {b.Shape[0]}.");

        int n = b.Shape[1];
        if (a.Rank == 2)
        {
            int M = a.Shape[0];
            var c = new Tensor<float>(M, n);
            GpuAccelerator.AcceleratedMatMul(a, b, c, target);

            c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (c.RequiresGrad)
            {
                AutogradTape.Current?.Record(new TapeEntry(AutogradOp.MatMul, c, a, b));
            }
            return c;
        }

        // Flatten leading batch dimensions [d0, ..., d_{r-2}, K] -> [M_flat, K]
        Span<int> outDims = stackalloc int[a.Rank];
        for (int i = 0; i < a.Rank - 1; i++)
        {
            outDims[i] = a.Shape[i];
        }
        outDims[a.Rank - 1] = n;
        var cNd = new Tensor<float>(new Shape8(outDims));

        long mFlatL = k > 0 ? a.ElementCount / k : 0;
        int mFlat = checked((int)mFlatL);
        if (mFlat > 0 && n > 0 && k > 0)
        {
            using var aContig = a.IsContiguous ? null : a.Contiguous();
            var aSrc = aContig ?? a;
            var shape2dA = new Shape8([mFlat, k]);
            var shape2dC = new Shape8([mFlat, n]);
            using var a2d = new Tensor<float>(aSrc.MemoryBlock, shape2dA, Strides8.ComputeContiguous(shape2dA), aSrc.ElementOffset, addRef: true);
            using var c2d = new Tensor<float>(cNd.MemoryBlock, shape2dC, Strides8.ComputeContiguous(shape2dC), cNd.ElementOffset, addRef: true);
            GpuAccelerator.AcceleratedMatMul(a2d, b, c2d, target);
        }

        cNd.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        if (cNd.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.MatMul, cNd, a, b));
        }
        return cNd;
    }

    public static Tensor<float> Add(Tensor<float> a, Tensor<float> b)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Add(a, b, c);

        c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Add, c, a, b));
        }
        return c;
    }

    public static Tensor<float> Subtract(Tensor<float> a, Tensor<float> b)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Subtract(a, b, c);

        c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Subtract, c, a, b));
        }
        return c;
    }

    public static Tensor<float> Multiply(Tensor<float> a, Tensor<float> b)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Multiply(a, b, c);

        c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Multiply, c, a, b));
        }
        return c;
    }

    public static Tensor<float> Scale(Tensor<float> a, float scalar)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Scale(a, scalar, c);

        c.RequiresGrad = a.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Scale, c, a, null, scalar));
        }
        return c;
    }

    public static Tensor<float> ReLU(Tensor<float> a)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.ReLU(a, c);

        c.RequiresGrad = a.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.ReLU, c, a));
        }
        return c;
    }

    public static Tensor<float> Sigmoid(Tensor<float> a)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Sigmoid(a, c);

        c.RequiresGrad = a.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Sigmoid, c, a));
        }
        return c;
    }

    public static Tensor<float> GELU(Tensor<float> a)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.GELU(a, c);

        c.RequiresGrad = a.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.GELU, c, a));
        }
        return c;
    }

    public static Tensor<float> RMSNorm(Tensor<float> a, Tensor<float>? weight = null, float eps = 1e-5f)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.RMSNorm(a, weight, c, eps);

        c.RequiresGrad = a.RequiresGrad || (weight?.RequiresGrad == true);
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.RMSNorm, c, a, weight, eps));
        }
        return c;
    }

    public static Tensor<float> Softmax(Tensor<float> a, int dim = -1)
    {
        int normalizedDim = dim < 0 ? dim + a.Rank : dim;
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Softmax(a, c, normalizedDim);

        c.RequiresGrad = a.RequiresGrad;
        if (c.RequiresGrad)
        {
            AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Softmax, c, a, intParam0: normalizedDim));
        }
        return c;
    }
}
