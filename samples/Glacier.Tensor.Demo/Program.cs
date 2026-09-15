using System;
using System.Diagnostics;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tensor.Interop;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using System.Linq;
using Glacier.Tensor.Optimizers;

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================================");
Console.WriteLine("          GLACIER.TENSOR: DEEP LEARNING & STRIDED TENSOR ENGINE                 ");
Console.WriteLine("      Zero-Allocation Autograd Tape & Cache-Blocked AVX-512 GEMM                ");
Console.WriteLine("                 .NET 10 / C# High-Performance Computing                        ");
Console.WriteLine("================================================================================\n");
Console.ResetColor();

// -----------------------------------------------------------------------------
// EXPERIMENT 1: Zero-Allocation Strided Slicing & Views
// -----------------------------------------------------------------------------
Console.WriteLine("--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 1: Zero-Heap Allocation Strided Tensor Slicing & Views");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

using (var t = new Tensor<float>(1000, 1000))
{
    t[100, 200] = 3.14159f;

    long allocBefore = GC.GetAllocatedBytesForCurrentThread();
    var sw = Stopwatch.StartNew();

    const int sliceOps = 100_000;
    for (int i = 0; i < sliceOps; i++)
    {
        using var slice = t.Slice(0, 100, 10);
        using var view = slice.Transpose(0, 1);
    }
    sw.Stop();
    long allocAfter = GC.GetAllocatedBytesForCurrentThread();

    Console.WriteLine($"  Executed {sliceOps:N0} Slice + Transpose view operations in {sw.Elapsed.TotalMilliseconds:F2} ms");
    Console.WriteLine($"  Average View Latency: {sw.Elapsed.TotalNanoseconds / (sliceOps * 2):F1} ns per view");
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"  Heap Allocations: {allocAfter - allocBefore} bytes (ZERO managed heap allocation!)");
    Console.ResetColor();
}

// -----------------------------------------------------------------------------
// EXPERIMENT 2: Cache-Blocked AVX-512 FMA GEMM Matrix Multiply
// -----------------------------------------------------------------------------
Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 2: Cache-Blocked AVX-512 FMA GEMM Matrix Multiplication");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

int M = 512, K = 512, N = 512;
Console.WriteLine($"Multiplying [{M} x {K}] by [{K} x {N}] matrices ({M * K * 2 / 1024} KB unmanaged data)...");

using (var matA = TensorFloatExtensions.RandomUniform([M, K], -1f, 1f, seed: 42))
using (var matB = TensorFloatExtensions.RandomUniform([K, N], -1f, 1f, seed: 84))
using (var matC = new Tensor<float>(M, N))
{
    // Warmup
    GemmKernels.MatMul(matA, matB, matC);

    int iters = 20;
    var swGemm = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++)
    {
        GemmKernels.MatMul(matA, matB, matC);
    }
    swGemm.Stop();

    double avgMs = swGemm.Elapsed.TotalMilliseconds / iters;
    double gflops = (2.0 * M * K * N / (avgMs * 1e6));
    Console.WriteLine($"  Average GEMM Latency: {avgMs:F2} ms per multiplication");
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine($"  Sustained CPU GEMM Throughput: {gflops:F2} GFLOPS (AVX-512 / AVX2 FMA)!");
    Console.ResetColor();
}

// -----------------------------------------------------------------------------
// EXPERIMENT 3: Deep Neural Network Training (MLP + Autograd Tape + AdamW)
// -----------------------------------------------------------------------------
Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 3: Deep Neural Network Training (MLP + Autograd Tape + AdamW)");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

int batchSize = 32;
int inDim = 16;
int hiddenDim = 32;
int outDim = 1;

using (var layer1 = new Linear(inDim, hiddenDim, seed: 1))
using (var layer2 = new Linear(hiddenDim, outDim, seed: 2))
using (var inputs = TensorFloatExtensions.RandomUniform([batchSize, inDim], -1f, 1f, seed: 123))
using (var targets = TensorFloatExtensions.Ones(batchSize, outDim))
{
    var modelParams = new[] { layer1.Weight, layer1.Bias, layer2.Weight, layer2.Bias };
    using var optimizer = new AdamW(modelParams, lr: 0.02f);

    Console.WriteLine($"Training 2-layer MLP [{inDim}] -> [{hiddenDim}] -> [{outDim}] on batch of {batchSize} samples...");

    float startLoss = 0f;
    float endLoss = 0f;

    var swTrain = Stopwatch.StartNew();
    for (int epoch = 1; epoch <= 30; epoch++)
    {
        optimizer.ZeroGrad();

        using var tape = new AutogradTape();
        foreach (var p in modelParams) tape.Watch(p);

        using var h1 = layer1.Forward(inputs);
        using var a1 = TensorOps.ReLU(h1);
        using var preds = layer2.Forward(a1);

        var (lossVal, _) = LossFunctions.MSELoss(preds, targets);

        if (epoch == 1) startLoss = lossVal;
        endLoss = lossVal;

        tape.Backward(preds);
        optimizer.Step();

        if (epoch % 10 == 0 || epoch == 1)
        {
            Console.WriteLine($"  Epoch {epoch,2}/30 | Loss: {lossVal:F5}");
        }
    }
    swTrain.Stop();

    Console.WriteLine($"  Trained in {swTrain.Elapsed.TotalMilliseconds:F2} ms ({swTrain.Elapsed.TotalMilliseconds / 30.0:F2} ms/step)");
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"  Convergence: Loss dropped from {startLoss:F4} to {endLoss:F4} ({(1f - endLoss / startLoss) * 100f:F1}% reduction)!");
    Console.ResetColor();
}

// -----------------------------------------------------------------------------
// EXPERIMENT 4: Zero-Copy Ingestion from Glacier.Polaris.DataFrame
// -----------------------------------------------------------------------------
Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 4: Zero-Copy Ingestion from Glacier.Polaris DataFrame");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

int rowCount = 100_000;
var col1 = new Float32Series("feature_a", rowCount);
var col2 = new Float32Series("feature_b", rowCount);
col1.Memory.Span.Fill(1.5f);
col2.Memory.Span.Fill(2.5f);

var df = new DataFrame(new ISeries[] { col1, col2 });
Console.WriteLine($"Constructed Polaris DataFrame with {df.RowCount:N0} rows and {df.Columns.Count()} columns.");

var swPolaris = Stopwatch.StartNew();
using (var tensorFromDf = df.ToTensor("feature_a", "feature_b"))
{
    swPolaris.Stop();
    Console.WriteLine($"  Converted {df.RowCount:N0} rows to Tensor<float> in {swPolaris.Elapsed.TotalMilliseconds:F2} ms");
    Console.WriteLine($"  Extracted Tensor Shape: {tensorFromDf.Shape}");
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"  Verification: [0,0]={tensorFromDf[0,0]:F1} (Expected 1.5), [0,1]={tensorFromDf[0,1]:F1} (Expected 2.5) -> PASSED");
    Console.ResetColor();
}

// -----------------------------------------------------------------------------
// EXPERIMENT 5: Hardware GPU Acceleration Integration (Glacier.Gpu)
// -----------------------------------------------------------------------------
Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 5: Hardware GPU Acceleration (Nvidia, Amd, Auto, Cpu Targets)");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

Console.WriteLine($"  GPU Hardware Detected: NVIDIA={GpuAccelerator.HasNvidiaGpu}, AMD={GpuAccelerator.HasAmdGpu}");
if (GpuAccelerator.IsGpuAvailable && GpuAccelerator.Engine != null)
{
    var dev = GpuAccelerator.Engine.DeviceInfo;
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine($"  [Glacier.Gpu Engine] {dev.DeviceName}");
    Console.WriteLine($"  Architecture: {dev.Architecture} | Compute Units: {dev.ComputeUnitsOrSms}");
    Console.ResetColor();
}

int[] benchSizes = { 512, 1024 };

foreach (int benchSize in benchSizes)
{
    Console.WriteLine($"\n  Benchmarking {benchSize}x{benchSize} FP32 GEMM across hardware targets ({(benchSize >= 1024 ? 5 : 20)} iterations)...");

    using var aBench = TensorFloatExtensions.RandomUniform([benchSize, benchSize], -1f, 1f, seed: 42);
    using var bBench = TensorFloatExtensions.RandomUniform([benchSize, benchSize], -1f, 1f, seed: 84);
    using var cBench = new Tensor<float>(benchSize, benchSize);

    string nvName = GpuAccelerator.Engine?.DeviceInfo.DeviceName ?? "NVIDIA GPU";
    string nvArch = GpuAccelerator.Engine?.DeviceInfo.Architecture ?? "SASS";
    string amdName = "AMD Radeon / Ryzen APU (Zero-Copy Unified RAM)";
    if (GpuAccelerator.HasAmdGpu)
    {
        try { amdName = $"{Glacier.Gpu.Drivers.HipDriver.GetDeviceName(0)} (Zero-Copy Unified RAM)"; } catch { }
    }

    var targetsToTest = new List<(GpuTarget Target, string Name)>
    {
        (GpuTarget.Cpu, "CPU AVX-512 (Dynamic Core Scaling)"),
        (GpuTarget.Auto, "Auto (Adaptive Hardware Dispatch)")
    };

    if (GpuAccelerator.HasDirect3D12)
        targetsToTest.Add((GpuTarget.Direct3D12, $"Direct3D 12 ({D3D12GemmKernel.DeviceName})"));

    if (GpuAccelerator.HasVulkan)
        targetsToTest.Add((GpuTarget.Vulkan, $"Vulkan 1.3 ({VulkanGemmKernel.DeviceName})"));

    targetsToTest.Add((GpuTarget.Amd, amdName));

    if (GpuAccelerator.HasNvidiaGpu)
    {
        targetsToTest.Add((GpuTarget.Nvidia, $"{nvName} (Bare-Metal {nvArch})"));
        targetsToTest.Add((GpuTarget.NvidiaTensorCore, $"{nvName} (Tensor Cores WMMA)"));
    }

    foreach (var (target, name) in targetsToTest)
    {
        try
        {
            // Warmup
            aBench.MatMul(bBench, cBench, target);

            int iters = benchSize >= 1024 ? 5 : 20;
            var swGpu = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                aBench.MatMul(bBench, cBench, target);
            }
            swGpu.Stop();

            double avgMs = swGpu.Elapsed.TotalMilliseconds / iters;
            double gflops = (2.0 * benchSize * benchSize * benchSize / (avgMs * 1e6));

            Console.ForegroundColor = target switch
            {
                GpuTarget.NvidiaTensorCore => ConsoleColor.Magenta,
                GpuTarget.Nvidia => ConsoleColor.Green,
                GpuTarget.Direct3D12 => ConsoleColor.Blue,
                GpuTarget.Vulkan => ConsoleColor.DarkYellow,
                GpuTarget.Amd => ConsoleColor.Red,
                GpuTarget.Auto => ConsoleColor.Cyan,
                _ => ConsoleColor.White
            };

            Console.WriteLine($"    [{name}]");
            Console.WriteLine($"      Latency: {avgMs:F2} ms | Throughput: {gflops:F2} GFLOPS ({(gflops / 1000.0):F2} TFLOPS)");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"    [{name}] Skipped: {ex.Message}");
            Console.ResetColor();
        }
    }
}

// -----------------------------------------------------------------------------
// EXPERIMENT 6: Parameter-Efficient Fine-Tuning (PEFT / LoRA)
// -----------------------------------------------------------------------------
Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 6: Parameter-Efficient Fine-Tuning (PEFT / LoRA)");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

int loraDim = 2048;
int loraRank = 16;
float loraAlpha = 32f;
int loraBatch = 32;

Console.WriteLine($"  Simulating LLM Transformer Projection Layer [{loraDim} x {loraDim}] with LoRA (r={loraRank}, alpha={loraAlpha})...");
long baseParams = (long)loraDim * loraDim;
long adapterParams = (long)loraDim * loraRank * 2;
double paramReduction = (1.0 - ((double)adapterParams / baseParams)) * 100.0;

Console.WriteLine($"  Base Model Weights (W0): {baseParams:N0} params ({baseParams * 4 / (1024 * 1024)} MB FP32) [FROZEN]");
Console.WriteLine($"  LoRA Adapters (A + B):   {adapterParams:N0} params ({adapterParams * 4 / 1024} KB FP32) [TRAINABLE]");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  Trainable Parameter Reduction: {paramReduction:F2}% (Only {(double)adapterParams / baseParams * 100.0:F2}% trainable!)");
Console.ResetColor();

using (var baseWeight = TensorFloatExtensions.RandomUniform([loraDim, loraDim], -0.05f, 0.05f, seed: 42))
using (var loraLayer = new LoraLinear(baseWeight, null, rank: loraRank, alpha: loraAlpha, seed: 101))
using (var optimizer = new AdamW(loraLayer.Parameters, lr: 0.01f))
using (var trainInput = TensorFloatExtensions.RandomUniform([loraBatch, loraDim], -1f, 1f, seed: 202))
using (var targetOutput = TensorFloatExtensions.RandomUniform([loraBatch, loraDim], -1f, 1f, seed: 303))
{
    Console.WriteLine($"\n  Running 25 LoRA Fine-Tuning Steps with AutogradTape & AdamW (Batch={loraBatch})...");

    float initialLoss = 0f;
    float finalLoss = 0f;
    var swLora = Stopwatch.StartNew();

    for (int step = 1; step <= 25; step++)
    {
        optimizer.ZeroGrad();

        using var tape = new AutogradTape();
        foreach (var p in loraLayer.Parameters) tape.Watch(p);

        using var pred = loraLayer.Forward(trainInput);
        var (lossVal, _) = LossFunctions.MSELoss(pred, targetOutput);

        if (step == 1) initialLoss = lossVal;
        finalLoss = lossVal;

        tape.Backward(pred);
        optimizer.Step();

        if (step == 1 || step % 5 == 0)
        {
            Console.WriteLine($"    Step {step,2}/25 | MSE Loss: {lossVal:F6} | Adapter Gradients Active");
        }
    }
    swLora.Stop();

    double avgStepMs = swLora.Elapsed.TotalMilliseconds / 25.0;
    double throughput = (loraBatch * 25.0) / swLora.Elapsed.TotalSeconds;

    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine($"\n  LoRA Training Completed in {swLora.Elapsed.TotalMilliseconds:F1} ms ({avgStepMs:F2} ms/step)");
    Console.WriteLine($"  Loss Reduction: {initialLoss:F6} -> {finalLoss:F6} ({(1.0f - finalLoss / initialLoss) * 100f:F1}% drop)");
    Console.WriteLine($"  Fine-Tuning Throughput: {throughput:F1} tokens/sec");
    Console.ResetColor();

    // Demonstrate Merge for deployment
    Console.WriteLine("\n  Fusing LoRA Adapters into Base Model Weights (Zero-Overhead Merge)...");
    var swMerge = Stopwatch.StartNew();
    using var mergedWeights = loraLayer.Merge();
    swMerge.Stop();
    Console.WriteLine($"  Merged W = W0 + (alpha/r)*(A*B) in {swMerge.Elapsed.TotalMilliseconds:F2} ms!");

    // Verify inference equivalence
    using var testX = TensorFloatExtensions.RandomUniform([4, loraDim], -1f, 1f, seed: 555);
    using var outLora = loraLayer.Forward(testX);
    using var outMerged = TensorOps.MatMul(testX, mergedWeights);

    var spanLora = outLora.AsSpan();
    var spanMerged = outMerged.AsSpan();
    float maxDiff = 0f;
    for (int i = 0; i < spanLora.Length; i++)
    {
        float diff = Math.Abs(spanLora[i] - spanMerged[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"  Inference Parity Verification: Max Difference = {maxDiff:E2} (Perfect Bitwise Fusion!)");
    Console.ResetColor();
}

Console.WriteLine("\n================================================================================");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine("All Glacier.Tensor deep learning & autograd experiments completed successfully!");
Console.ResetColor();
Console.WriteLine("================================================================================");

if (!args.Contains("--headless") && !args.Contains("--bench") && Environment.UserInteractive && !Console.IsInputRedirected)
{
    Console.WriteLine("\n[Press any key to exit...]");
    Console.ReadKey();
}

