using System;
using Glacier.Tensor.Compute;

namespace Glacier.Tensor.Core;

/// <summary>
/// Specialized high-performance mathematical and statistical initializers for floating-point tensors.
/// </summary>
public static class TensorFloatExtensions
{
    public static Tensor<float> Ones(params ReadOnlySpan<int> shape)
    {
        var t = new Tensor<float>(shape);
        t.Fill(1.0f);
        return t;
    }

    public static Tensor<float> RandomUniform(ReadOnlySpan<int> shape, float min = -1.0f, float max = 1.0f, int? seed = null)
    {
        var t = new Tensor<float>(shape);
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        var span = t.AsSpan();
        float diff = max - min;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = min + (float)rng.NextDouble() * diff;
        }
        return t;
    }

    public static Tensor<float> RandomNormal(ReadOnlySpan<int> shape, float mean = 0.0f, float std = 1.0f, int? seed = null)
    {
        var t = new Tensor<float>(shape);
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        var span = t.AsSpan();

        // Box-Muller transform
        for (int i = 0; i < span.Length; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            span[i] = (float)(mean + std * randStdNormal);

            if (i + 1 < span.Length)
            {
                double randStdNormal2 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                span[i + 1] = (float)(mean + std * randStdNormal2);
            }
        }
        return t;
    }

    public static Tensor<float> KaimingUniform(ReadOnlySpan<int> shape, int fanIn, int? seed = null)
    {
        float bound = (float)Math.Sqrt(3.0 / Math.Max(1, fanIn));
        return RandomUniform(shape, -bound, bound, seed);
    }

    /// <summary>
    /// Executes matrix multiplication C = A * B using specified hardware target (Auto, Nvidia, Amd, DualGpu, or Cpu).
    /// </summary>
    public static void MatMul(this Tensor<float> a, Tensor<float> b, Tensor<float> c, GpuTarget target = GpuTarget.Auto)
    {
        GpuAccelerator.AcceleratedMatMul(a, b, c, target);
    }

    /// <summary>
    /// Executes matrix multiplication C = A * B using specified hardware target (Auto, Nvidia, Amd, DualGpu, or Cpu), allocating a new result tensor.
    /// </summary>
    public static Tensor<float> MatMul(this Tensor<float> a, Tensor<float> b, GpuTarget target = GpuTarget.Auto)
    {
        return TensorOps.MatMul(a, b, target);
    }
}
