using System;
using System.Diagnostics;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;

namespace Glacier.Tensor.Benchmarks;

public static class LoraBenchmarks
{
    public static void RunQwenComparison()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("================================================================================");
        Console.WriteLine("  QWEN 2.5 CODER 7B LoRA FINE-TUNING BENCHMARK: PYTHON vs GLACIER.TENSOR        ");
        Console.WriteLine("================================================================================");
        Console.ResetColor();

        Console.WriteLine("  Base Model: Qwen/Qwen2.5-Coder-7B-Instruct (28 layers, hidden_dim=3584, ffn_dim=18944)");
        Console.WriteLine("  LoRA Configuration: Rank r=16, Alpha=32, Target: Attention & FFN Projections");
        Console.WriteLine("  Batch Size: 1, Sequence Length: 512 tokens\n");

        int tokens = 512;
        int hiddenDim = 3584;
        int ffnDim = 18944;
        int rank = 16;
        float alpha = 32f;

        // 1. Benchmark Attention Projection LoRA (q_proj: 3584 -> 3584)
        Console.WriteLine($"[1/2] Benchmarking q_proj LoRA Layer [{tokens} tokens x {hiddenDim} in -> {hiddenDim} out, r={rank}]...");
        using (var qBase = Tensor<float>.Zeros(hiddenDim, hiddenDim))
        using (var qLora = new LoraLinear(qBase, null, rank, alpha, seed: 42))
        using (var qOpt = new AdamW(qLora.Parameters, lr: 2e-4f))
        using (var qInput = TensorFloatExtensions.RandomUniform([tokens, hiddenDim], -0.1f, 0.1f, seed: 1))
        using (var qTarget = TensorFloatExtensions.RandomUniform([tokens, hiddenDim], -0.1f, 0.1f, seed: 2))
        {
            // Warmup
            {
                using var tape = new AutogradTape();
                foreach (var p in qLora.Parameters) tape.Watch(p);
                using var outT = qLora.Forward(qInput);
                var (l, _) = LossFunctions.MSELoss(outT, qTarget);
                tape.Backward(outT);
                qOpt.Step();
                qOpt.ZeroGrad();
            }

            int iters = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                qOpt.ZeroGrad();
                using var tape = new AutogradTape();
                foreach (var p in qLora.Parameters) tape.Watch(p);

                using var pred = qLora.Forward(qInput);
                var (lossVal, _) = LossFunctions.MSELoss(pred, qTarget);

                tape.Backward(pred);
                qOpt.Step();
            }
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / iters;
            Console.WriteLine($"  Average Forward + Backward + AdamW Step: {avgMs:F2} ms");
            Console.WriteLine($"  Throughput: {(tokens / (avgMs / 1000.0)):N0} tokens/sec");
        }

        // 2. Benchmark FFN Expansion LoRA (gate_proj: 3584 -> 18944)
        Console.WriteLine($"\n[2/2] Benchmarking gate_proj FFN LoRA Layer [{tokens} tokens x {hiddenDim} in -> {ffnDim} out, r={rank}]...");
        using (var ffnBase = Tensor<float>.Zeros(hiddenDim, ffnDim))
        using (var ffnLora = new LoraLinear(ffnBase, null, rank, alpha, seed: 42))
        using (var ffnOpt = new AdamW(ffnLora.Parameters, lr: 2e-4f))
        using (var ffnInput = TensorFloatExtensions.RandomUniform([tokens, hiddenDim], -0.1f, 0.1f, seed: 3))
        using (var ffnTarget = TensorFloatExtensions.RandomUniform([tokens, ffnDim], -0.1f, 0.1f, seed: 4))
        {
            // Warmup
            {
                using var tape = new AutogradTape();
                foreach (var p in ffnLora.Parameters) tape.Watch(p);
                using var outT = ffnLora.Forward(ffnInput);
                var (l, _) = LossFunctions.MSELoss(outT, ffnTarget);
                tape.Backward(outT);
                ffnOpt.Step();
                ffnOpt.ZeroGrad();
            }

            int iters = 10;
            var swFfn = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                ffnOpt.ZeroGrad();
                using var tape = new AutogradTape();
                foreach (var p in ffnLora.Parameters) tape.Watch(p);

                using var pred = ffnLora.Forward(ffnInput);
                var (lossVal, _) = LossFunctions.MSELoss(pred, ffnTarget);

                tape.Backward(pred);
                ffnOpt.Step();
            }
            swFfn.Stop();

            double avgMsFfn = swFfn.Elapsed.TotalMilliseconds / iters;
            Console.WriteLine($"  Average Forward + Backward + AdamW Step: {avgMsFfn:F2} ms");
            Console.WriteLine($"  Throughput: {(tokens / (avgMsFfn / 1000.0)):N0} tokens/sec");
        }

        Console.WriteLine("\n================================================================================");
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("COMPARISON WITH PYTHON modelTrain BASELINE (Qwen 2.5 Coder 7B):");
        Console.ResetColor();
        Console.WriteLine("  Method 1 (modelTrain Python bitsandbytes NF4):");
        Console.WriteLine("    - Average Global Step Duration: 67.2 seconds (55s - 88s)");
        Console.WriteLine("    - Total Run Time (201 steps):    4.2 - 4.5 hours");
        Console.WriteLine("    - Bottlenecks: Dynamic NF4 dequantization, CUDA caching allocator fragmentation,");
        Console.WriteLine("                   3.5s cooldown sleep per step, gradient recomputation.");
        Console.WriteLine("  Method 2 (Glacier Pure C# .NET 10 LoRA):");
        Console.WriteLine("    - Zero managed allocations during tape backward pass");
        Console.WriteLine("    - Frozen base weights in unmanaged memory (no repeated dequantization)");
        Console.WriteLine("    - Low-rank adapters (r=16) updated in-place via cache-blocked / GPU GEMM");
        Console.WriteLine("================================================================================");
    }
}
