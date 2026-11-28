using System;
using System.Collections.Generic;
using System.Diagnostics;
using BenchmarkDotNet.Running;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Benchmarks;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;

Console.WriteLine("Glacier.Tensor Benchmarks Runner");
Console.WriteLine("================================");

if (args.Length > 0 && args[0].Equals("--bdn", StringComparison.OrdinalIgnoreCase))
{
    BenchmarkRunner.Run<GemmBenchmarks>();
}
else
{
    Console.WriteLine("Running high-precision GEMM microbenchmarks...\n");

    int[] sizes = { 256, 512, 1024, 2048 };
    foreach (int size in sizes)
    {
        using var a = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 1);
        using var b = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 2);
        using var c = new Tensor<float>(size, size);

        // Warmup
        GemmKernels.MatMul(a, b, c);

        int iters = size >= 2048 ? 3 : (size >= 1024 ? 5 : 20);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            GemmKernels.MatMul(a, b, c);
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / iters;
        double gflops = (2.0 * size * size * size / (avgMs * 1e6));
        Console.WriteLine($"[CPU GEMM {size} x {size}] Average: {avgMs:F2} ms | Throughput: {gflops:F2} GFLOPS");
    }

    Console.WriteLine("\nRunning AutogradTape + Sequential + AdamW step benchmark...");
    RunAutogradTrainingBenchmark();

    Console.WriteLine("\nAll CPU GEMM & Autograd microbenchmarks finished.\n");
    LoraBenchmarks.RunQwenComparison();
    GpuBenchmarks.RunGpuComparison();
    GpuBenchmarks.RunD3D12Benchmark();
    GpuBenchmarks.RunVulkanBenchmark();
}

static void RunAutogradTrainingBenchmark()
{
    const int batch = 64;
    const int inFeat = 128;
    const int hidden = 256;
    const int classes = 16;

    using var l1 = new Linear(inFeat, hidden, seed: 1);
    using var norm = new LayerNorm(hidden);
    using var l2 = new Linear(hidden, classes, seed: 2);
    var parameters = new List<Tensor<float>> { l1.Weight, l1.Bias, norm.Gamma, norm.Beta, l2.Weight, l2.Bias };
    using var optimizer = new AdamW(parameters, lr: 1e-3f);
    using var x = TensorFloatExtensions.RandomUniform([batch, inFeat], -1f, 1f, seed: 7);
    using var targets = Tensor<float>.Zeros(batch, classes);
    for (int i = 0; i < batch; i++) targets[i, i % classes] = 1.0f;

    // Warmup
    for (int w = 0; w < 5; w++)
    {
        optimizer.ZeroGrad();
        using var tape = new AutogradTape();
        foreach (var p in parameters) tape.Watch(p);
        using var h1 = l1.Forward(x);
        using var n1 = norm.Forward(h1);
        using var act = TensorOps.GELU(n1);
        using var logits = l2.Forward(act);
        var (_, lossT) = LossFunctions.CrossEntropy(logits, targets);
        using var _loss = lossT;
        tape.Backward(lossT);
        optimizer.Step();
    }

    const int steps = 50;
    long totalBackwardAlloc = 0;
    var sw = Stopwatch.StartNew();
    for (int s = 0; s < steps; s++)
    {
        optimizer.ZeroGrad();
        using var tape = new AutogradTape();
        foreach (var p in parameters) tape.Watch(p);
        using var h1 = l1.Forward(x);
        using var n1 = norm.Forward(h1);
        using var act = TensorOps.GELU(n1);
        using var logits = l2.Forward(act);
        var (_, lossT) = LossFunctions.CrossEntropy(logits, targets);
        using var _loss = lossT;
        long beforeBwd = GC.GetAllocatedBytesForCurrentThread();
        tape.Backward(lossT);
        totalBackwardAlloc += GC.GetAllocatedBytesForCurrentThread() - beforeBwd;
        optimizer.Step();
    }
    sw.Stop();
    Console.WriteLine($"[Autograd MLP 64x128->256->16 (50 steps)] Avg Step: {sw.Elapsed.TotalMilliseconds / steps:F3} ms | Backward Managed Alloc: {totalBackwardAlloc} B total ({totalBackwardAlloc / steps} B/step)");
}

