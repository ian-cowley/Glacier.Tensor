using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;
using Glacier.Tensor.Benchmarks;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

Console.WriteLine("Glacier.Tensor Benchmarks Runner");
Console.WriteLine("================================");

if (args.Length > 0 && args[0].Equals("--bdn", StringComparison.OrdinalIgnoreCase))
{
    BenchmarkRunner.Run<GemmBenchmarks>();
}
else
{
    Console.WriteLine("Running high-precision GEMM microbenchmarks...\n");

    int[] sizes = { 256, 512, 1024 };
    foreach (int size in sizes)
    {
        using var a = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 1);
        using var b = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 2);
        using var c = new Tensor<float>(size, size);

        // Warmup
        GemmKernels.MatMul(a, b, c);

        int iters = size >= 1024 ? 5 : 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            GemmKernels.MatMul(a, b, c);
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / iters;
        double gflops = (2.0 * size * size * size / (avgMs * 1e6));
        Console.WriteLine($"[GEMM {size} x {size}] Average: {avgMs:F2} ms | Throughput: {gflops:F2} GFLOPS (AVX-512 FMA)");
    }

    Console.WriteLine("\nAll benchmarks finished successfully.");
}
