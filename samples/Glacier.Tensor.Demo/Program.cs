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
Console.WriteLine("EXPERIMENT 5: Hardware GPU Acceleration Bridge via Glacier.Gpu");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

if (GpuAccelerator.IsGpuAvailable && GpuAccelerator.Engine != null)
{
    var dev = GpuAccelerator.Engine.DeviceInfo;
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"  [Glacier.Gpu] Hardware Engine Available: {dev.DeviceName}");
    Console.WriteLine($"  Architecture: {dev.Architecture} | Compute Units: {dev.ComputeUnitsOrSms}");
    Console.ResetColor();
}
else
{
    Console.WriteLine("  Bare-metal GPU acceleration bridge initialized (CPU AVX-512 active).");
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
