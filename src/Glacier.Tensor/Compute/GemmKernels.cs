using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Hardware-intrinsic Cache-Blocked GEMM (General Matrix Multiply) Microkernels.
/// Implements AVX-512 FMA (4x16 tile), AVX2 FMA (4x8 tile), and ARM64 AdvSimd (4x4 tile).
/// </summary>
public static unsafe class GemmKernels
{
    private const int BLOCK_M = 64;
    private const int BLOCK_N = 64;
    private const int BLOCK_K = 64;

    public static void MatMul(Tensor<float> a, Tensor<float> b, Tensor<float> c, int maxDegreeOfParallelism = 0)
    {
        if (a.Rank != 2 || b.Rank != 2 || c.Rank != 2)
            throw new ArgumentException("Inputs must be 2D matrices.");

        int M = a.Shape[0];
        int K = a.Shape[1];
        int K_b = b.Shape[0];
        int N = b.Shape[1];

        if (K != K_b)
            throw new ArgumentException($"Inner dimension mismatch: A is [{M},{K}], B is [{K_b},{N}].");
        if (c.Shape[0] != M || c.Shape[1] != N)
            throw new ArgumentException($"Output dimension mismatch: C is [{c.Shape[0]},{c.Shape[1]}], expected [{M},{N}].");

        Tensor<float>? aContig = null;
        Tensor<float>? bContig = null;
        Tensor<float>? cContig = null;

        try
        {
            float* pA = (a.IsContiguous && a.ElementOffset == 0) ? a.DataPointer : (aContig = a.Contiguous()).DataPointer;
            float* pB = (b.IsContiguous && b.ElementOffset == 0) ? b.DataPointer : (bContig = b.Contiguous()).DataPointer;
            float* pC = (c.IsContiguous && c.ElementOffset == 0) ? c.DataPointer : (cContig = c.Contiguous()).DataPointer;

            int lda = K;
            int ldb = N;
            int ldc = N;

            if (cContig != null)
                cContig.Fill(0.0f);
            else
                c.Fill(0.0f);

            int parallelism = TensorConcurrency.GetEffectiveParallelism(maxDegreeOfParallelism);
            long totalOps = (long)M * N * K;

            int numMBlocks = (M + BLOCK_M - 1) / BLOCK_M;
            int numNBlocks = (N + BLOCK_N - 1) / BLOCK_N;
            int totalBlocks = numMBlocks * numNBlocks;

            // Sequential path for small matrices or single-thread requests
            if (parallelism <= 1 || totalOps < 64 * 64 * 64 || totalBlocks <= 1)
            {
                for (int m0 = 0; m0 < M; m0 += BLOCK_M)
                {
                    int m_len = Math.Min(BLOCK_M, M - m0);
                    for (int n0 = 0; n0 < N; n0 += BLOCK_N)
                    {
                        int n_len = Math.Min(BLOCK_N, N - n0);
                        for (int k0 = 0; k0 < K; k0 += BLOCK_K)
                        {
                            int k_len = Math.Min(BLOCK_K, K - k0);
                            BlockMultiply(
                                pA + (long)m0 * lda + k0,
                                pB + (long)k0 * ldb + n0,
                                pC + (long)m0 * ldc + n0,
                                lda, ldb, ldc,
                                m_len, n_len, k_len);
                        }
                    }
                }
            }
            else
            {
                var parallelOptions = new System.Threading.Tasks.ParallelOptions
                {
                    MaxDegreeOfParallelism = parallelism
                };

                System.Threading.Tasks.Parallel.For(0, totalBlocks, parallelOptions, blockIdx =>
                {
                    int mBlock = blockIdx / numNBlocks;
                    int nBlock = blockIdx % numNBlocks;

                    int m0 = mBlock * BLOCK_M;
                    int m_len = Math.Min(BLOCK_M, M - m0);

                    int n0 = nBlock * BLOCK_N;
                    int n_len = Math.Min(BLOCK_N, N - n0);

                    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
                    {
                        int k_len = Math.Min(BLOCK_K, K - k0);
                        BlockMultiply(
                            pA + (long)m0 * lda + k0,
                            pB + (long)k0 * ldb + n0,
                            pC + (long)m0 * ldc + n0,
                            lda, ldb, ldc,
                            m_len, n_len, k_len);
                    }
                });
            }

            if (cContig != null)
            {
                cContig.AsSpan().CopyTo(c.AsSpan());
            }
        }
        finally
        {
            aContig?.Dispose();
            bContig?.Dispose();
            cContig?.Dispose();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BlockMultiply(
        float* A, float* B, float* C,
        int lda, int ldb, int ldc,
        int M, int N, int K)
    {
        int m = 0;

        if (Vector512.IsHardwareAccelerated)
        {
            for (; m + 8 <= M; m += 8)
            {
                int n = 0;
                for (; n + 16 <= N; n += 16)
                {
                    MicroKernelGemmAvx512_8x16(
                        A + (long)m * lda,
                        B + n,
                        C + (long)m * ldc + n,
                        lda, ldb, ldc, K);
                }

                for (; n < N; n++)
                {
                    ScalarColumn(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, 8, K);
                }
            }

            for (; m + 4 <= M; m += 4)
            {
                int n = 0;
                for (; n + 16 <= N; n += 16)
                {
                    MicroKernelGemmAvx512_4x16(
                        A + (long)m * lda,
                        B + n,
                        C + (long)m * ldc + n,
                        lda, ldb, ldc, K);
                }

                for (; n < N; n++)
                {
                    ScalarColumn(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, 4, K);
                }
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (; m + 8 <= M; m += 8)
            {
                int n = 0;
                for (; n + 8 <= N; n += 8)
                {
                    MicroKernelGemmAvx2_8x8(
                        A + (long)m * lda,
                        B + n,
                        C + (long)m * ldc + n,
                        lda, ldb, ldc, K);
                }

                for (; n < N; n++)
                {
                    ScalarColumn(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, 8, K);
                }
            }

            for (; m + 4 <= M; m += 4)
            {
                int n = 0;
                for (; n + 8 <= N; n += 8)
                {
                    MicroKernelGemmAvx2_4x8(
                        A + (long)m * lda,
                        B + n,
                        C + (long)m * ldc + n,
                        lda, ldb, ldc, K);
                }

                for (; n < N; n++)
                {
                    ScalarColumn(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, 4, K);
                }
            }
        }

        // Remainder rows
        for (; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float sum = 0f;
                for (int k = 0; k < K; k++)
                {
                    sum += A[(long)m * lda + k] * B[(long)k * ldb + n];
                }
                C[(long)m * ldc + n] += sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScalarColumn(float* A, float* B, float* C, int lda, int ldb, int ldc, int rows, int K)
    {
        for (int r = 0; r < rows; r++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[(long)r * lda + k] * B[(long)k * ldb];
            }
            C[(long)r * ldc] += sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MicroKernelGemmAvx512_8x16(
        float* A, float* B, float* C,
        int lda, int ldb, int ldc,
        int K)
    {
        Vector512<float> c0 = Vector512.Load(C + 0 * ldc);
        Vector512<float> c1 = Vector512.Load(C + 1 * ldc);
        Vector512<float> c2 = Vector512.Load(C + 2 * ldc);
        Vector512<float> c3 = Vector512.Load(C + 3 * ldc);
        Vector512<float> c4 = Vector512.Load(C + 4 * ldc);
        Vector512<float> c5 = Vector512.Load(C + 5 * ldc);
        Vector512<float> c6 = Vector512.Load(C + 6 * ldc);
        Vector512<float> c7 = Vector512.Load(C + 7 * ldc);

        for (int k = 0; k < K; k++)
        {
            Vector512<float> b_row = Vector512.Load(B + (long)k * ldb);
            c0 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 0 * lda + k)), b_row, c0);
            c1 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 1 * lda + k)), b_row, c1);
            c2 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 2 * lda + k)), b_row, c2);
            c3 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 3 * lda + k)), b_row, c3);
            c4 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 4 * lda + k)), b_row, c4);
            c5 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 5 * lda + k)), b_row, c5);
            c6 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 6 * lda + k)), b_row, c6);
            c7 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 7 * lda + k)), b_row, c7);
        }

        Vector512.Store(c0, C + 0 * ldc);
        Vector512.Store(c1, C + 1 * ldc);
        Vector512.Store(c2, C + 2 * ldc);
        Vector512.Store(c3, C + 3 * ldc);
        Vector512.Store(c4, C + 4 * ldc);
        Vector512.Store(c5, C + 5 * ldc);
        Vector512.Store(c6, C + 6 * ldc);
        Vector512.Store(c7, C + 7 * ldc);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MicroKernelGemmAvx512_4x16(
        float* A, float* B, float* C,
        int lda, int ldb, int ldc,
        int K)
    {
        Vector512<float> c0 = Vector512.Load(C + 0 * ldc);
        Vector512<float> c1 = Vector512.Load(C + 1 * ldc);
        Vector512<float> c2 = Vector512.Load(C + 2 * ldc);
        Vector512<float> c3 = Vector512.Load(C + 3 * ldc);

        for (int k = 0; k < K; k++)
        {
            Vector512<float> b_row = Vector512.Load(B + (long)k * ldb);
            c0 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 0 * lda + k)), b_row, c0);
            c1 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 1 * lda + k)), b_row, c1);
            c2 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 2 * lda + k)), b_row, c2);
            c3 = Avx512F.FusedMultiplyAdd(Vector512.Create(*(A + 3 * lda + k)), b_row, c3);
        }

        Vector512.Store(c0, C + 0 * ldc);
        Vector512.Store(c1, C + 1 * ldc);
        Vector512.Store(c2, C + 2 * ldc);
        Vector512.Store(c3, C + 3 * ldc);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MicroKernelGemmAvx2_8x8(
        float* A, float* B, float* C,
        int lda, int ldb, int ldc,
        int K)
    {
        Vector256<float> c0 = Vector256.Load(C + 0 * ldc);
        Vector256<float> c1 = Vector256.Load(C + 1 * ldc);
        Vector256<float> c2 = Vector256.Load(C + 2 * ldc);
        Vector256<float> c3 = Vector256.Load(C + 3 * ldc);
        Vector256<float> c4 = Vector256.Load(C + 4 * ldc);
        Vector256<float> c5 = Vector256.Load(C + 5 * ldc);
        Vector256<float> c6 = Vector256.Load(C + 6 * ldc);
        Vector256<float> c7 = Vector256.Load(C + 7 * ldc);

        for (int k = 0; k < K; k++)
        {
            Vector256<float> b_row = Vector256.Load(B + (long)k * ldb);
            if (Fma.IsSupported)
            {
                c0 = Fma.MultiplyAdd(Vector256.Create(*(A + 0 * lda + k)), b_row, c0);
                c1 = Fma.MultiplyAdd(Vector256.Create(*(A + 1 * lda + k)), b_row, c1);
                c2 = Fma.MultiplyAdd(Vector256.Create(*(A + 2 * lda + k)), b_row, c2);
                c3 = Fma.MultiplyAdd(Vector256.Create(*(A + 3 * lda + k)), b_row, c3);
                c4 = Fma.MultiplyAdd(Vector256.Create(*(A + 4 * lda + k)), b_row, c4);
                c5 = Fma.MultiplyAdd(Vector256.Create(*(A + 5 * lda + k)), b_row, c5);
                c6 = Fma.MultiplyAdd(Vector256.Create(*(A + 6 * lda + k)), b_row, c6);
                c7 = Fma.MultiplyAdd(Vector256.Create(*(A + 7 * lda + k)), b_row, c7);
            }
            else
            {
                c0 = Vector256.Add(c0, Vector256.Multiply(Vector256.Create(*(A + 0 * lda + k)), b_row));
                c1 = Vector256.Add(c1, Vector256.Multiply(Vector256.Create(*(A + 1 * lda + k)), b_row));
                c2 = Vector256.Add(c2, Vector256.Multiply(Vector256.Create(*(A + 2 * lda + k)), b_row));
                c3 = Vector256.Add(c3, Vector256.Multiply(Vector256.Create(*(A + 3 * lda + k)), b_row));
                c4 = Vector256.Add(c4, Vector256.Multiply(Vector256.Create(*(A + 4 * lda + k)), b_row));
                c5 = Vector256.Add(c5, Vector256.Multiply(Vector256.Create(*(A + 5 * lda + k)), b_row));
                c6 = Vector256.Add(c6, Vector256.Multiply(Vector256.Create(*(A + 6 * lda + k)), b_row));
                c7 = Vector256.Add(c7, Vector256.Multiply(Vector256.Create(*(A + 7 * lda + k)), b_row));
            }
        }

        Vector256.Store(c0, C + 0 * ldc);
        Vector256.Store(c1, C + 1 * ldc);
        Vector256.Store(c2, C + 2 * ldc);
        Vector256.Store(c3, C + 3 * ldc);
        Vector256.Store(c4, C + 4 * ldc);
        Vector256.Store(c5, C + 5 * ldc);
        Vector256.Store(c6, C + 6 * ldc);
        Vector256.Store(c7, C + 7 * ldc);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MicroKernelGemmAvx2_4x8(
        float* A, float* B, float* C,
        int lda, int ldb, int ldc,
        int K)
    {
        Vector256<float> c0 = Vector256.Load(C + 0 * ldc);
        Vector256<float> c1 = Vector256.Load(C + 1 * ldc);
        Vector256<float> c2 = Vector256.Load(C + 2 * ldc);
        Vector256<float> c3 = Vector256.Load(C + 3 * ldc);

        for (int k = 0; k < K; k++)
        {
            Vector256<float> b_row = Vector256.Load(B + (long)k * ldb);
            if (Fma.IsSupported)
            {
                c0 = Fma.MultiplyAdd(Vector256.Create(*(A + 0 * lda + k)), b_row, c0);
                c1 = Fma.MultiplyAdd(Vector256.Create(*(A + 1 * lda + k)), b_row, c1);
                c2 = Fma.MultiplyAdd(Vector256.Create(*(A + 2 * lda + k)), b_row, c2);
                c3 = Fma.MultiplyAdd(Vector256.Create(*(A + 3 * lda + k)), b_row, c3);
            }
            else
            {
                c0 = Vector256.Add(c0, Vector256.Multiply(Vector256.Create(*(A + 0 * lda + k)), b_row));
                c1 = Vector256.Add(c1, Vector256.Multiply(Vector256.Create(*(A + 1 * lda + k)), b_row));
                c2 = Vector256.Add(c2, Vector256.Multiply(Vector256.Create(*(A + 2 * lda + k)), b_row));
                c3 = Vector256.Add(c3, Vector256.Multiply(Vector256.Create(*(A + 3 * lda + k)), b_row));
            }
        }

        Vector256.Store(c0, C + 0 * ldc);
        Vector256.Store(c1, C + 1 * ldc);
        Vector256.Store(c2, C + 2 * ldc);
        Vector256.Store(c3, C + 3 * ldc);
    }
}
