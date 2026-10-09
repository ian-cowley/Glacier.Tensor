using System;
using System.Diagnostics;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Benchmarks;

public static class GpuBenchmarks
{
    public static void RunGpuComparison()
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("================================================================================");
        Console.WriteLine("  NVIDIA RTX 4060 GPU GEMM: SYNCHRONOUS PCIE COPY vs ZERO-COPY RESIDENT         ");
        Console.WriteLine("================================================================================");
        Console.ResetColor();

        if (!GpuAccelerator.HasNvidiaGpu)
        {
            Console.WriteLine("  [SKIP] NVIDIA GPU not detected or CUDA driver unavailable.");
            return;
        }

        int[] sizes = { 1024, 2048 };
        foreach (int size in sizes)
        {
            Console.WriteLine($"\n--- GEMM Size {size} x {size} ---");

            using var aHost = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 1);
            using var bHost = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 2);
            using var cHost = new Tensor<float>(size, size);

            // 1. Host Copied GEMM (Includes HtoD + Kernel + DtoH each invocation)
            // Warmup
            GpuAccelerator.ExecuteNvidiaGemm(aHost, bHost, cHost);

            int iters = 20;
            var swCopied = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                GpuAccelerator.ExecuteNvidiaGemm(aHost, bHost, cHost);
            }
            swCopied.Stop();
            double avgMsCopied = swCopied.Elapsed.TotalMilliseconds / iters;
            double gflopsCopied = (2.0 * size * size * size / (avgMsCopied * 1e6));

            // 2. Device Resident GEMM (Zero PCIe copies during compute loop via DeviceResidentTensor)
            using var aDev = DeviceResidentTensor<float>.FromHost(aHost);
            using var bDev = DeviceResidentTensor<float>.FromHost(bHost);
            using var cDev = DeviceResidentTensor<float>.Allocate(size, size);

            // Warmup
            aDev.MatMul(bDev, cDev);

            var swDev = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                aDev.MatMul(bDev, cDev);
            }
            swDev.Stop();
            using var cHostResult = cDev.ToHost();

            double avgMsDev = swDev.Elapsed.TotalMilliseconds / iters;
            double gflopsDev = (2.0 * size * size * size / (avgMsDev * 1e6));
            double speedup = avgMsCopied / avgMsDev;

            Console.WriteLine($"  [Copied GEMM]          Latency: {avgMsCopied:F3} ms | Throughput: {gflopsCopied:F1} GFLOPS");
            Console.WriteLine($"  [Device-Resident GEMM] Latency: {avgMsDev:F3} ms | Throughput: {gflopsDev:F1} GFLOPS ({gflopsDev / 1000.0:F2} TFLOPS)");
            Console.WriteLine($"  -> Speedup: {speedup:F2}x lower latency bypassing PCIe HtoD/DtoH overhead");
        }
    }

    public static void RunD3D12Benchmark()
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("\n================================================================================");
        Console.WriteLine("  DIRECT3D 12 COMPUTE GEMM: 64x64 TILING & PERSISTENT GPU HEAP ACCELERATION    ");
        Console.WriteLine("================================================================================");
        Console.ResetColor();

        if (!D3D12GemmKernel.IsSupported)
        {
            Console.WriteLine("  [SKIP] Direct3D 12 compute is not supported on this platform/adapter.");
            return;
        }

        Console.WriteLine($"  Active D3D12 Hardware Adapter: {D3D12GemmKernel.DeviceName}");

        int[] sizes = { 1024, 2048 };
        foreach (int size in sizes)
        {
            using var a = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 1);
            using var b = TensorFloatExtensions.RandomUniform([size, size], -1f, 1f, seed: 2);
            using var c = new Tensor<float>(size, size);

            // Warmup
            D3D12GemmKernel.Execute(a, b, c);

            int iters = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                D3D12GemmKernel.Execute(a, b, c);
            }
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / iters;
            double gflops = (2.0 * size * size * size / (avgMs * 1e6));
            Console.WriteLine($"  [D3D12 Copied {size}x{size}]   Latency: {avgMs:F3} ms | Throughput: {gflops:F1} GFLOPS ({gflops / 1000.0:F2} TFLOPS)");

            // Resident D3D12 execution
            D3D12GemmKernel.UploadBuffers(a.AsSpan(), b.AsSpan(), size, size, size);
            D3D12GemmKernel.ExecuteResident(size, size, size);

            var swRes = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                D3D12GemmKernel.ExecuteResident(size, size, size);
            }
            swRes.Stop();

            double avgMsRes = swRes.Elapsed.TotalMilliseconds / iters;
            double gflopsRes = (2.0 * size * size * size / (avgMsRes * 1e6));
            double d3dSpeedup = avgMs / avgMsRes;
            Console.WriteLine($"  [D3D12 Resident {size}x{size}] Latency: {avgMsRes:F3} ms | Throughput: {gflopsRes:F1} GFLOPS ({gflopsRes / 1000.0:F2} TFLOPS)");
            Console.WriteLine($"  -> Speedup: {d3dSpeedup:F2}x lower latency bypassing D3D12 staging copies");
        }
    }
}
