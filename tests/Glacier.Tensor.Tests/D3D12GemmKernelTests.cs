using System;
using System.Collections.Generic;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tensor.Diagnostics;
using Xunit;

namespace Glacier.Tensor.Tests;

public class D3D12GemmKernelTests
{
    [Fact]
    public void Execute_NullTensors_ThrowsArgumentNullException()
    {
        using var a = Tensor<float>.Zeros(16, 16);
        using var b = Tensor<float>.Zeros(16, 16);
        using var c = Tensor<float>.Zeros(16, 16);

        // Overload with out diagnostics
        Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(null!, b, c, out _));
        Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, null!, c, out _));
        Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, b, null!, out _));

        // Overload without diagnostics
        Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(null!, b, c));
        Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, null!, c));
        Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, b, null!));
    }

    [Fact]
    public void Execute_Non2DTensors_FailsAndPopulatesDiagnostics()
    {
        D3D12GemmKernel.ClearLastError();

        using var a1D = new Tensor<float>(16);
        using var b2D = Tensor<float>.Zeros(16, 16);
        using var c2D = Tensor<float>.Zeros(16, 16);

        bool success = D3D12GemmKernel.Execute(a1D, b2D, c2D, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.False(diag.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Equal("ArgumentValidation", diag.StageName);
        Assert.Contains("Rank=2", diag.ErrorMessage);
        Assert.Same(diag, D3D12GemmKernel.LastError);
    }

    [Fact]
    public void Execute_InnerDimensionMismatch_FailsAndPopulatesDiagnostics()
    {
        D3D12GemmKernel.ClearLastError();

        // A is [16, 32], B is [16, 64] -> inner dim mismatch: 32 != 16
        using var a = Tensor<float>.Zeros(16, 32);
        using var b = Tensor<float>.Zeros(16, 64);
        using var c = Tensor<float>.Zeros(16, 64);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Contains("Inner matrix dimensions mismatch", diag.ErrorMessage);
    }

    [Fact]
    public void Execute_ResultDimensionMismatch_FailsAndPopulatesDiagnostics()
    {
        D3D12GemmKernel.ClearLastError();

        // A is [16, 32], B is [32, 64] -> product is [16, 64], but C is [16, 16]
        using var a = Tensor<float>.Zeros(16, 32);
        using var b = Tensor<float>.Zeros(32, 64);
        using var c = Tensor<float>.Zeros(16, 16);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Contains("Result tensor shape", diag.ErrorMessage);
    }

    [Fact]
    public void Execute_NonContiguousTensor_FailsAndPopulatesDiagnostics()
    {
        D3D12GemmKernel.ClearLastError();

        // Transposed matrix is non-contiguous
        using var aOriginal = new Tensor<float>(16, 32);
        using var aTransposed = aOriginal.Transpose();
        Assert.False(aTransposed.IsContiguous);

        using var b = Tensor<float>.Zeros(16, 32);
        using var c = Tensor<float>.Zeros(32, 32);

        bool success = D3D12GemmKernel.Execute(aTransposed, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Contains("contiguous", diag.ErrorMessage, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Execute_LogsErrorToGlacierDiagnostics()
    {
        var loggedEntries = new List<(LogLevel Level, string Message, Exception? Ex)>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
        {
            loggedEntries.Add((level, msg, ex));
        });

        try
        {
            using var a = Tensor<float>.Zeros(16, 32);
            using var b = Tensor<float>.Zeros(16, 64);
            using var c = Tensor<float>.Zeros(16, 64);

            bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);
            Assert.False(success);

            Assert.NotEmpty(loggedEntries);
            Assert.Contains(loggedEntries, e => e.Level == LogLevel.Error && e.Message.Contains("ArgumentValidation"));
        }
        finally
        {
            GlacierDiagnostics.Reset();
        }
    }

    [Fact]
    public void ClearLastError_ResetsDiagnosticsState()
    {
        using var a = Tensor<float>.Zeros(16, 32);
        using var b = Tensor<float>.Zeros(16, 64);
        using var c = Tensor<float>.Zeros(16, 64);

        D3D12GemmKernel.Execute(a, b, c, out _);
        Assert.NotNull(D3D12GemmKernel.LastError);

        D3D12GemmKernel.ClearLastError();
        Assert.Null(D3D12GemmKernel.LastError);
        Assert.Null(D3D12GemmKernel.LastException);
    }

    [Fact]
    public void Execute_HardwareExecutionOrGracefulFallback()
    {
        D3D12GemmKernel.ClearLastError();

        int M = 64, K = 64, N = 64;
        using var a = TensorFloatExtensions.RandomUniform([M, K], -1.0f, 1.0f, seed: 1234);
        using var b = TensorFloatExtensions.RandomUniform([K, N], -1.0f, 1.0f, seed: 5678);
        using var c = new Tensor<float>(M, N);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        if (D3D12GemmKernel.IsSupported)
        {
            Assert.True(success);
            Assert.NotNull(diag);
            Assert.True(diag.Success);
            Assert.Equal(D3D12ExecutionStage.None, diag.Stage);
            Assert.Null(D3D12GemmKernel.LastError);

            // Verify accuracy against CPU computation
            using var cCpu = new Tensor<float>(M, N);
            a.MatMul(b, cCpu, GpuTarget.Cpu);

            var spanGpu = c.AsSpan();
            var spanCpu = cCpu.AsSpan();
            Assert.Equal(spanCpu.Length, spanGpu.Length);

            for (int i = 0; i < spanCpu.Length; i++)
            {
                Assert.True(MathF.Abs(spanGpu[i] - spanCpu[i]) < 1e-3f,
                    $"Mismatch at index {i}: GPU={spanGpu[i]}, CPU={spanCpu[i]}");
            }
        }
        else
        {
            Assert.False(success);
            Assert.NotNull(diag);
            Assert.False(diag.Success);
            Assert.Equal(D3D12ExecutionStage.Initialization, diag.Stage);
        }
    }

    [Fact]
    public void D3D12KernelDiagnostics_Properties_Accessible()
    {
        var diag = new D3D12KernelDiagnostics
        {
            Success = true,
            Stage = D3D12ExecutionStage.Dispatch,
            DeviceName = "Test Adapter",
            IsDeviceRemoved = false,
            DeviceRemovedReasonHResult = 0,
            DeviceRemovedReasonDescription = "OK",
            M = 64,
            K = 64,
            N = 64,
            BytesA = 1024,
            BytesB = 1024,
            BytesC = 1024,
            GridX = 1,
            GridY = 1,
            ErrorMessage = "Test Error",
            ExceptionType = typeof(InvalidOperationException).FullName,
            HResult = -1
        };

        Assert.True(diag.Success);
        Assert.Equal(D3D12ExecutionStage.Dispatch, diag.Stage);
        Assert.Equal("Dispatch", diag.StageName);
        Assert.Equal("Test Adapter", diag.DeviceName);
        Assert.Equal("Test Error", diag.ErrorMessage);
        Assert.Equal("Test Error", diag.ExceptionMessage);

        diag.ExceptionMessage = "Updated Error";
        Assert.Equal("Updated Error", diag.ErrorMessage);
    }
}
