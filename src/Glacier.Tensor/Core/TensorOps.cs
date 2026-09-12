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
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMul expects 2D tensors.");

        int M = a.Shape[0];
        int N = b.Shape[1];
        var c = new Tensor<float>(M, N);
        GpuAccelerator.AcceleratedMatMul(a, b, c, target);

        c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.MatMul, c, a, b));
        return c;
    }

    public static Tensor<float> Add(Tensor<float> a, Tensor<float> b)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Add(a, b, c);

        c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Add, c, a, b));
        return c;
    }

    public static Tensor<float> Subtract(Tensor<float> a, Tensor<float> b)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Subtract(a, b, c);

        c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Subtract, c, a, b));
        return c;
    }

    public static Tensor<float> Multiply(Tensor<float> a, Tensor<float> b)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Multiply(a, b, c);

        c.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Multiply, c, a, b));
        return c;
    }

    public static Tensor<float> Scale(Tensor<float> a, float scalar)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Scale(a, scalar, c);

        c.RequiresGrad = a.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Scale, c, a, null, scalar));
        return c;
    }

    public static Tensor<float> ReLU(Tensor<float> a)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.ReLU(a, c);

        c.RequiresGrad = a.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.ReLU, c, a));
        return c;
    }

    public static Tensor<float> Sigmoid(Tensor<float> a)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Sigmoid(a, c);

        c.RequiresGrad = a.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.Sigmoid, c, a));
        return c;
    }

    public static Tensor<float> GELU(Tensor<float> a)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.GELU(a, c);

        c.RequiresGrad = a.RequiresGrad;
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.GELU, c, a));
        return c;
    }

    public static Tensor<float> RMSNorm(Tensor<float> a, Tensor<float>? weight = null, float eps = 1e-5f)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.RMSNorm(a, weight, c, eps);

        c.RequiresGrad = a.RequiresGrad || (weight?.RequiresGrad == true);
        AutogradTape.Current?.Record(new TapeEntry(AutogradOp.RMSNorm, c, a, weight, eps));
        return c;
    }

    public static Tensor<float> Softmax(Tensor<float> a, int dim = -1)
    {
        var c = new Tensor<float>(a.Shape);
        ElementwiseKernels.Softmax(a, c, dim);
        c.RequiresGrad = a.RequiresGrad;
        return c;
    }
}
