using System;
using System.Runtime.InteropServices;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tensor.Diagnostics;
using Xunit;

namespace Glacier.Tensor.Tests.StressChallenge;

/// <summary>
/// Empirical adversarial stress test harness for Milestone M4 challenger validation.
/// Tests extreme/adversarial boundary conditions on NativeMemoryBlock<T>.AsSpan
/// and comprehensive error telemetry on D3D12GemmKernel.
/// </summary>
public unsafe class AdversarialChallengerM4Tests
{
    // =========================================================================
    // SECTION 1: NativeMemoryBlock<T>.AsSpan Adversarial Bounds Stress Tests
    // =========================================================================

    [Theory]
    [InlineData(-1L)]
    [InlineData(-100L)]
    [InlineData(long.MinValue)]
    [InlineData(-2L)]
    [InlineData(-999999L)]
    [InlineData(long.MinValue + 1)]
    public void NativeMemoryBlock_NegativeOffsets_ThrowsArgumentOutOfRangeException(long negativeOffset)
    {
        using var block = new NativeMemoryBlock<float>(500);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(negativeOffset));
        Assert.Equal("offset", ex.ParamName);
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(-100)]
    [InlineData(int.MinValue)]
    [InlineData(-3)]
    [InlineData(-999999)]
    [InlineData(int.MinValue + 1)]
    public void NativeMemoryBlock_NegativeLengths_ThrowsArgumentOutOfRangeException(int negativeLength)
    {
        using var block = new NativeMemoryBlock<float>(500);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(0, negativeLength));
        Assert.Equal("length", ex.ParamName);
    }

    [Theory]
    [InlineData(501L)]     // count + 1
    [InlineData(1500L)]    // count + 1000
    [InlineData(1000000L)] // far beyond capacity
    [InlineData(long.MaxValue)]
    public void NativeMemoryBlock_OffsetExceedingCapacity_ThrowsArgumentOutOfRangeException(long excessiveOffset)
    {
        const long capacity = 500;
        using var block = new NativeMemoryBlock<float>(capacity);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(excessiveOffset, 0));
        Assert.Equal("offset", ex.ParamName);
    }

    [Fact]
    public void NativeMemoryBlock_Overrun_OffsetCountMinus5_Length10_ThrowsArgumentOutOfRangeException()
    {
        const long capacity = 100;
        using var block = new NativeMemoryBlock<float>(capacity);

        // Requested: offset = count - 5 = 95, length = 10 -> overruns capacity by 5 elements
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(95, 10));
        Assert.Equal("length", ex.ParamName);
    }

    [Theory]
    [InlineData(95L, 6)]   // count - 5 with length 6 (1 element overrun)
    [InlineData(99L, 2)]   // count - 1 with length 2 (1 element overrun)
    [InlineData(100L, 1)]  // offset at end with length 1 (1 element overrun)
    [InlineData(0L, 101)]  // offset 0 with length count + 1
    [InlineData(50L, 51)]  // offset 50 with length 51
    public void NativeMemoryBlock_VariousOverruns_ThrowArgumentOutOfRangeException(long offset, int length)
    {
        const long capacity = 100;
        using var block = new NativeMemoryBlock<float>(capacity);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(offset, length));
        Assert.Equal("length", ex.ParamName);
    }

    [Fact]
    public void NativeMemoryBlock_64BitArithmeticOverflow_OffsetLongMaxValueMinus10_Length20()
    {
        const long capacity = 100;
        using var block = new NativeMemoryBlock<float>(capacity);

        // In 64-bit signed arithmetic, (long.MaxValue - 10) + 20 wraps around to long.MinValue + 9.
        // A naive check `offset + length > capacity` would wrap around and bypass bounds checking.
        // The implementation must reject this invalid offset cleanly.
        long overflowOffset = long.MaxValue - 10;
        int length = 20;

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(overflowOffset, length));
        Assert.Equal("offset", ex.ParamName);
    }

    [Theory]
    [InlineData(long.MaxValue, 1)]
    [InlineData(long.MaxValue - 1, 2)]
    [InlineData(long.MaxValue - 100, 200)]
    public void NativeMemoryBlock_ExtremeOverflowOffsets_RejectCleanly(long offset, int length)
    {
        using var block = new NativeMemoryBlock<float>(50);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(offset, length));
        Assert.Equal("offset", ex.ParamName);
    }

    [Fact]
    public void NativeMemoryBlock_IntMaxValueLengthFromPositiveOffset_ThrowsArgumentOutOfRangeException()
    {
        using var block = new NativeMemoryBlock<float>(100);

        // offset 50, length int.MaxValue exceeds remaining capacity (50)
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(50, int.MaxValue));
        Assert.Equal("length", ex.ParamName);
    }

    [Fact]
    public void NativeMemoryBlock_DisposedMemoryBlock_ThrowsObjectDisposedException()
    {
        var block = new NativeMemoryBlock<float>(200);
        Assert.False(block.IsDisposed);

        block.Dispose();
        Assert.True(block.IsDisposed);

        // Default AsSpan()
        Assert.Throws<ObjectDisposedException>(() => block.AsSpan());

        // AsSpan with parameters (positive, zero, negative)
        Assert.Throws<ObjectDisposedException>(() => block.AsSpan(0, 10));
        Assert.Throws<ObjectDisposedException>(() => block.AsSpan(0, 0));
        Assert.Throws<ObjectDisposedException>(() => block.AsSpan(-1, 5));
        Assert.Throws<ObjectDisposedException>(() => block.AsSpan(500, 10));

        // AddRef on disposed block
        Assert.Throws<ObjectDisposedException>(() => block.AddRef());

        // Double dispose should be idempotent and not throw
        block.Dispose();
        Assert.True(block.IsDisposed);
    }

    [Fact]
    public void NativeMemoryBlock_ZeroCapacityBlock_SlicingBehaviors()
    {
        using var block = new NativeMemoryBlock<float>(0);
        Assert.Equal(0, block.ElementCount);
        Assert.Equal(0u, block.ByteLength);
        Assert.True(block.Pointer == null);

        // Valid slices on 0-capacity block:
        var spanDefault = block.AsSpan();
        Assert.True(spanDefault.IsEmpty);
        Assert.Equal(0, spanDefault.Length);

        var spanExplicitZero = block.AsSpan(0, 0);
        Assert.True(spanExplicitZero.IsEmpty);
        Assert.Equal(0, spanExplicitZero.Length);

        var spanSentinel = block.AsSpan(0, -1);
        Assert.True(spanSentinel.IsEmpty);
        Assert.Equal(0, spanSentinel.Length);

        // Invalid slices on 0-capacity block:
        var exOffset = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(1, 0));
        Assert.Equal("offset", exOffset.ParamName);

        var exNegativeOffset = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(-1, 0));
        Assert.Equal("offset", exNegativeOffset.ParamName);

        var exLength = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(0, 1));
        Assert.Equal("length", exLength.ParamName);

        var exNegativeLength = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(0, -2));
        Assert.Equal("length", exNegativeLength.ParamName);
    }

    [Fact]
    public void NativeMemoryBlock_ValidSlices_SucceedAndRetainDataIntegrity()
    {
        const int count = 256;
        using var block = new NativeMemoryBlock<float>(count);

        var fullSpan = block.AsSpan();
        Assert.Equal(count, fullSpan.Length);

        // Fill with pattern
        for (int i = 0; i < count; i++)
        {
            fullSpan[i] = (float)(i * 3.14159);
        }

        // Subslice interior
        var subSpan = block.AsSpan(64, 128);
        Assert.Equal(128, subSpan.Length);
        Assert.Equal((float)(64 * 3.14159), subSpan[0]);
        Assert.Equal((float)(191 * 3.14159), subSpan[127]);

        // Subslice to the end using default length (-1)
        var remSpan = block.AsSpan(200);
        Assert.Equal(56, remSpan.Length);
        Assert.Equal((float)(200 * 3.14159), remSpan[0]);
        Assert.Equal((float)(255 * 3.14159), remSpan[55]);

        // Boundary slice at the very end with length 0
        var emptyEnd = block.AsSpan(count, 0);
        Assert.True(emptyEnd.IsEmpty);

        var emptySentinel = block.AsSpan(count, -1);
        Assert.True(emptySentinel.IsEmpty);
    }

    [Fact]
    public void NativeMemoryBlock_EmpiricalFuzzingHarness_ValidatesAllEdgePermutations()
    {
        const long capacity = 1000;
        using var block = new NativeMemoryBlock<float>(capacity);

        var rng = new Random(4242);

        for (int iteration = 0; iteration < 500; iteration++)
        {
            // Pick offsets across negative, in-bounds, boundary, out-of-bounds, and extreme overflow
            long offset = (iteration % 5) switch
            {
                0 => rng.Next(-10000, -1),                          // Negative
                1 => rng.Next(0, (int)capacity + 1),                 // In bounds [0, 1000]
                2 => (long)capacity + rng.Next(1, 5000),             // Exceeding capacity
                3 => long.MaxValue - rng.Next(0, 1000),             // 64-bit overflow zone
                _ => (long)rng.Next(0, 50)                          // Low in bounds
            };

            // Pick lengths across negative, sentinel -1, zero, in-bounds, overrun, and extreme
            int length = (iteration % 6) switch
            {
                0 => rng.Next(-5000, -2),                            // Invalid negative
                1 => -1,                                             // Sentinel
                2 => 0,                                              // Zero length
                3 => rng.Next(1, (int)capacity * 2),                 // Positive
                4 => int.MinValue + rng.Next(0, 100),                // Negative extreme
                _ => int.MaxValue - rng.Next(0, 100)                 // Positive extreme
            };

            // Oracle truth calculation:
            bool expectOffsetError = offset < 0 || offset > capacity;
            bool expectLengthError = !expectOffsetError && length != -1 && (length < 0 || length > (capacity - offset));

            if (expectOffsetError)
            {
                var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(offset, length));
                Assert.Equal("offset", ex.ParamName);
            }
            else if (expectLengthError)
            {
                var ex = Assert.Throws<ArgumentOutOfRangeException>(() => block.AsSpan(offset, length));
                Assert.Equal("length", ex.ParamName);
            }
            else
            {
                var span = block.AsSpan(offset, length);
                int expectedLength = length == -1 ? (int)(capacity - offset) : length;
                Assert.Equal(expectedLength, span.Length);
            }
        }
    }

    // =========================================================================
    // SECTION 2: D3D12GemmKernel Adversarial Telemetry & Robustness Stress Tests
    // =========================================================================

    [Fact]
    public void D3D12GemmKernel_NullTensors_ThrowsArgumentNullExceptionWithAccurateParamName()
    {
        using var a = Tensor<float>.Zeros(16, 16);
        using var b = Tensor<float>.Zeros(16, 16);
        using var c = Tensor<float>.Zeros(16, 16);

        // Null 'a'
        var exAOut = Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(null!, b, c, out _));
        Assert.Equal("a", exAOut.ParamName);
        var exA = Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(null!, b, c));
        Assert.Equal("a", exA.ParamName);

        // Null 'b'
        var exBOut = Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, null!, c, out _));
        Assert.Equal("b", exBOut.ParamName);
        var exB = Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, null!, c));
        Assert.Equal("b", exB.ParamName);

        // Null 'c'
        var exCOut = Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, b, null!, out _));
        Assert.Equal("c", exCOut.ParamName);
        var exC = Assert.Throws<ArgumentNullException>(() => D3D12GemmKernel.Execute(a, b, null!));
        Assert.Equal("c", exC.ParamName);
    }

    [Theory]
    [InlineData(1)]  // 1D tensor
    [InlineData(3)]  // 3D tensor
    [InlineData(4)]  // 4D tensor
    public void D3D12GemmKernel_InvalidRank_TensorA_FailsGracefullyWithStructuredTelemetry(int invalidRank)
    {
        D3D12GemmKernel.ClearLastError();

        int[] shapeA = invalidRank switch
        {
            1 => [16],
            3 => [2, 16, 16],
            4 => [1, 2, 16, 16],
            _ => [16]
        };

        using var a = new Tensor<float>(shapeA);
        using var b = Tensor<float>.Zeros(16, 16);
        using var c = Tensor<float>.Zeros(16, 16);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.False(diag.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Equal("ArgumentValidation", diag.StageName);
        Assert.Contains("Tensor 'a' must be 2D matrix", diag.ErrorMessage);
        Assert.Equal(typeof(ArgumentException).FullName, diag.ExceptionType);
        Assert.NotNull(diag.Exception);
        Assert.Same(diag, D3D12GemmKernel.LastError);
    }

    [Theory]
    [InlineData(1)]  // 1D tensor
    [InlineData(3)]  // 3D tensor
    [InlineData(4)]  // 4D tensor
    public void D3D12GemmKernel_InvalidRank_TensorB_FailsGracefullyWithStructuredTelemetry(int invalidRank)
    {
        D3D12GemmKernel.ClearLastError();

        int[] shapeB = invalidRank switch
        {
            1 => [16],
            3 => [2, 16, 16],
            4 => [1, 2, 16, 16],
            _ => [16]
        };

        using var a = Tensor<float>.Zeros(16, 16);
        using var b = new Tensor<float>(shapeB);
        using var c = Tensor<float>.Zeros(16, 16);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.False(diag.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Equal("ArgumentValidation", diag.StageName);
        Assert.Contains("Tensor 'b' must be 2D matrix", diag.ErrorMessage);
        Assert.Equal(typeof(ArgumentException).FullName, diag.ExceptionType);
        Assert.NotNull(diag.Exception);
    }

    [Theory]
    [InlineData(1)]  // 1D tensor
    [InlineData(3)]  // 3D tensor
    [InlineData(4)]  // 4D tensor
    public void D3D12GemmKernel_InvalidRank_TensorC_FailsGracefullyWithStructuredTelemetry(int invalidRank)
    {
        D3D12GemmKernel.ClearLastError();

        int[] shapeC = invalidRank switch
        {
            1 => [16],
            3 => [2, 16, 16],
            4 => [1, 2, 16, 16],
            _ => [16]
        };

        using var a = Tensor<float>.Zeros(16, 16);
        using var b = Tensor<float>.Zeros(16, 16);
        using var c = new Tensor<float>(shapeC);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.False(diag.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Equal("ArgumentValidation", diag.StageName);
        Assert.Contains("Tensor 'c' must be 2D matrix", diag.ErrorMessage);
        Assert.Equal(typeof(ArgumentException).FullName, diag.ExceptionType);
        Assert.NotNull(diag.Exception);
    }

    [Theory]
    [InlineData(16, 32, 48, 64)] // A is [16, 32], B is [48, 64] -> K_A=32 != K_B=48
    [InlineData(64, 1, 2, 64)]   // A is [64, 1], B is [2, 64] -> K_A=1 != K_B=2
    [InlineData(10, 100, 99, 10)]// A is [10, 100], B is [99, 10] -> K_A=100 != K_B=99
    public void D3D12GemmKernel_InnerDimensionMismatch_FailsGracefullyWithStructuredTelemetry(
        int mA, int kA, int kB, int nB)
    {
        D3D12GemmKernel.ClearLastError();

        using var a = Tensor<float>.Zeros(mA, kA);
        using var b = Tensor<float>.Zeros(kB, nB);
        using var c = Tensor<float>.Zeros(mA, nB);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.False(diag.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Contains("Inner matrix dimensions mismatch", diag.ErrorMessage);
        Assert.Equal(typeof(ArgumentException).FullName, diag.ExceptionType);
    }

    [Theory]
    [InlineData(16, 32, 64, 8, 64)]   // C.Shape[0] = 8 != mA (16)
    [InlineData(16, 32, 64, 16, 32)]  // C.Shape[1] = 32 != nB (64)
    [InlineData(16, 32, 64, 32, 16)]  // Both C dimensions swapped
    [InlineData(64, 64, 64, 65, 64)]  // C slightly larger on M
    [InlineData(64, 64, 64, 64, 65)]  // C slightly larger on N
    public void D3D12GemmKernel_OutputShapeMismatch_FailsGracefullyWithStructuredTelemetry(
        int m, int k, int n, int cM, int cN)
    {
        D3D12GemmKernel.ClearLastError();

        using var a = Tensor<float>.Zeros(m, k);
        using var b = Tensor<float>.Zeros(k, n);
        using var c = Tensor<float>.Zeros(cM, cN);

        bool success = D3D12GemmKernel.Execute(a, b, c, out var diag);

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.False(diag.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Contains("Result tensor shape", diag.ErrorMessage);
        Assert.Equal(typeof(ArgumentException).FullName, diag.ExceptionType);
    }

    [Fact]
    public void D3D12GemmKernel_OverloadWithoutDiagnostics_SafelyPopulatesLastError()
    {
        D3D12GemmKernel.ClearLastError();
        Assert.Null(D3D12GemmKernel.LastError);

        using var a = Tensor<float>.Zeros(16, 32);
        using var b = Tensor<float>.Zeros(48, 64); // Inner dim mismatch
        using var c = Tensor<float>.Zeros(16, 64);

        // Invoke overload without out parameter
        bool success = D3D12GemmKernel.Execute(a, b, c);

        Assert.False(success);
        Assert.NotNull(D3D12GemmKernel.LastError);
        Assert.False(D3D12GemmKernel.LastError.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, D3D12GemmKernel.LastError.Stage);
        Assert.Contains("Inner matrix dimensions mismatch", D3D12GemmKernel.LastError.ErrorMessage);
    }

    [Fact]
    public void D3D12GemmKernel_StructuredDiagnosticsIntegrity_CapturesAllProperties()
    {
        D3D12GemmKernel.ClearLastError();

        using var a1D = new Tensor<float>(32);
        using var b2D = Tensor<float>.Zeros(32, 32);
        using var c2D = Tensor<float>.Zeros(32, 32);

        DateTimeOffset before = DateTimeOffset.UtcNow;
        bool success = D3D12GemmKernel.Execute(a1D, b2D, c2D, out var diag);
        DateTimeOffset after = DateTimeOffset.UtcNow;

        Assert.False(success);
        Assert.NotNull(diag);
        Assert.False(diag.Success);
        Assert.Equal(D3D12ExecutionStage.ArgumentValidation, diag.Stage);
        Assert.Equal("ArgumentValidation", diag.StageName);
        Assert.NotNull(diag.ErrorMessage);
        Assert.NotNull(diag.ExceptionType);
        Assert.NotNull(diag.Exception);
        Assert.NotNull(diag.StackTrace);
        Assert.True(diag.Timestamp >= before.AddSeconds(-1) && diag.Timestamp <= after.AddSeconds(1));
    }

    [Fact]
    public void D3D12GemmKernel_ConcurrentExecution_ExposesDiagnosticRaceCondition()
    {
        // Stress test concurrent calls to Execute with different error types.
        // Under thread safety, each caller's out diagnostics must match that caller's specific error.
        // If static LastError is clobbered across threads, out diagnostics will contain mismatched error messages.
        int iterations = 100;
        int mismatchCount = 0;

        Parallel.For(0, iterations, i =>
        {
            if (i % 2 == 0)
            {
                using var a1D = new Tensor<float>(16);
                using var b2D = Tensor<float>.Zeros(16, 16);
                using var c2D = Tensor<float>.Zeros(16, 16);
                D3D12GemmKernel.Execute(a1D, b2D, c2D, out var diag);
                if (diag != null && !diag.ErrorMessage!.Contains("Rank=2"))
                {
                    Interlocked.Increment(ref mismatchCount);
                }
            }
            else
            {
                using var a = Tensor<float>.Zeros(16, 32);
                using var b = Tensor<float>.Zeros(48, 64);
                using var c = Tensor<float>.Zeros(16, 64);
                D3D12GemmKernel.Execute(a, b, c, out var diag);
                if (diag != null && !diag.ErrorMessage!.Contains("Inner matrix dimensions mismatch"))
                {
                    Interlocked.Increment(ref mismatchCount);
                }
            }
        });

        Assert.True(mismatchCount == 0,
            $"Concurrency violation: {mismatchCount}/{iterations} concurrent D3D12GemmKernel.Execute calls received corrupt diagnostics from another thread due to static LastError sharing.");
    }
}
