using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Hardware-intrinsic Cache-Blocked GEMM (General Matrix Multiply) Microkernels.
/// Implements AVX-512 FMA (8x16 / 4x16 tile), AVX2 FMA (8x8 / 4x8 tile), ARM64 AdvSimd / Vector128 (4x4 tile),
/// and cache-friendly panel packing for contiguous and transposed operands.
/// </summary>
public static unsafe class GemmKernels
{
    private const int BLOCK_M = 64;
    private const int BLOCK_N = 64;
    private const int BLOCK_K = 64;

    [ThreadStatic]
    private static float[]? t_packA;

    [ThreadStatic]
    private static float[]? t_packB;

    private static float[] GetPackBufferA() => t_packA ??= new float[BLOCK_M * BLOCK_K];
    private static float[] GetPackBufferB() => t_packB ??= new float[BLOCK_K * BLOCK_N];

    public static void MatMul(Tensor<float> a, Tensor<float> b, Tensor<float> c, int maxDegreeOfParallelism = 0)
        => MatMul(a, b, c, transposeA: false, transposeB: false, maxDegreeOfParallelism);

    public static void MatMul(
        Tensor<float> a,
        Tensor<float> b,
        Tensor<float> c,
        bool transposeA,
        bool transposeB,
        int maxDegreeOfParallelism = 0)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        if (a.Rank != 2 || b.Rank != 2 || c.Rank != 2)
            throw new ArgumentException("Inputs must be 2D matrices.");

        int M = transposeA ? a.Shape[1] : a.Shape[0];
        int K = transposeA ? a.Shape[0] : a.Shape[1];
        int K_b = transposeB ? b.Shape[1] : b.Shape[0];
        int N = transposeB ? b.Shape[0] : b.Shape[1];

        if (K != K_b)
            throw new ArgumentException($"Inner dimension mismatch: op(A) is [{M},{K}], op(B) is [{K_b},{N}].");
        if (c.Shape[0] != M || c.Shape[1] != N)
            throw new ArgumentException($"Output dimension mismatch: C is [{c.Shape[0]},{c.Shape[1]}], expected [{M},{N}].");

        if (M == 0 || K == 0 || N == 0)
        {
            if (M > 0 && N > 0)
            {
                c.Fill(0.0f);
            }

            return;
        }

        Tensor<float>? aContig = null;
        Tensor<float>? bContig = null;
        Tensor<float>? cContig = null;

        try
        {
            float* pA = (a.IsContiguous && a.ElementOffset == 0) ? a.DataPointer : (aContig = a.Contiguous()).DataPointer;
            float* pB = (b.IsContiguous && b.ElementOffset == 0) ? b.DataPointer : (bContig = b.Contiguous()).DataPointer;
            float* pC = (c.IsContiguous && c.ElementOffset == 0) ? c.DataPointer : (cContig = c.Contiguous()).DataPointer;

            int ldaRaw = a.Shape[1];
            int ldbRaw = b.Shape[1];
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

            // Pack B (always if transposeB, or when B width exceeds cache-friendly row width or multiple M blocks)
            bool usePackB = transposeB || ldbRaw > BLOCK_N;

            if (parallelism <= 1 || totalOps < 64 * 64 * 64 || totalBlocks <= 1)
            {
                float[]? packABuf = transposeA ? GetPackBufferA() : null;
                float[]? packBBuf = usePackB ? GetPackBufferB() : null;

                fixed (float* pPackA = packABuf)
                fixed (float* pPackB = packBBuf)
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
                                ExecuteBlock(
                                    pA, pB, pC,
                                    ldaRaw, ldbRaw, ldc,
                                    m0, n0, k0, m_len, n_len, k_len,
                                    transposeA, transposeB, usePackB,
                                    pPackA, pPackB);
                            }
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

                    float[]? packABuf = transposeA ? GetPackBufferA() : null;
                    float[]? packBBuf = usePackB ? GetPackBufferB() : null;

                    fixed (float* pPackA = packABuf)
                    fixed (float* pPackB = packBBuf)
                    {
                        for (int k0 = 0; k0 < K; k0 += BLOCK_K)
                        {
                            int k_len = Math.Min(BLOCK_K, K - k0);
                            ExecuteBlock(
                                pA, pB, pC,
                                ldaRaw, ldbRaw, ldc,
                                m0, n0, k0, m_len, n_len, k_len,
                                transposeA, transposeB, usePackB,
                                pPackA, pPackB);
                        }
                    }
                });
            }

            if (cContig != null)
            {
                Tensor<float>.CopyElements(cContig, c);
            }
        }
        finally
        {
            aContig?.Dispose();
            bContig?.Dispose();
            cContig?.Dispose();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteBlock(
        float* pA,
        float* pB,
        float* pC,
        int ldaRaw,
        int ldbRaw,
        int ldc,
        int m0,
        int n0,
        int k0,
        int m_len,
        int n_len,
        int k_len,
        bool transposeA,
        bool transposeB,
        bool usePackB,
        float* pPackA,
        float* pPackB)
    {
        float* blockA;
        int lda;
        if (transposeA)
        {
            PackTransposedAPanel(pA, ldaRaw, pPackA, m0, k0, m_len, k_len);
            blockA = pPackA;
            lda = k_len;
        }
        else
        {
            blockA = pA + (long)m0 * ldaRaw + k0;
            lda = ldaRaw;
        }

        float* blockB;
        int ldb;
        if (usePackB)
        {
            PackBPanel(pB, ldbRaw, pPackB, k0, n0, k_len, n_len, transposeB);
            blockB = pPackB;
            ldb = n_len;
        }
        else
        {
            blockB = pB + (long)k0 * ldbRaw + n0;
            ldb = ldbRaw;
        }

        BlockMultiply(
            blockA,
            blockB,
            pC + (long)m0 * ldc + n0,
            lda, ldb, ldc,
            m_len, n_len, k_len);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PackTransposedAPanel(float* srcA, int ldaSrc, float* dst, int m0, int k0, int mLen, int kLen)
    {
        // A is stored as [K, M]; block is A^T[m0..m0+mLen, k0..k0+kLen] = A[k0..k0+kLen, m0..m0+mLen]
        for (int m = 0; m < mLen; m++)
        {
            float* dstRow = dst + (long)m * kLen;
            int colSrc = m0 + m;
            for (int k = 0; k < kLen; k++)
            {
                dstRow[k] = srcA[(long)(k0 + k) * ldaSrc + colSrc];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PackBPanel(float* srcB, int ldbSrc, float* dst, int k0, int n0, int kLen, int nLen, bool transposeB)
    {
        if (!transposeB)
        {
            for (int k = 0; k < kLen; k++)
            {
                Unsafe.CopyBlockUnaligned(
                    dst + (long)k * nLen,
                    srcB + (long)(k0 + k) * ldbSrc + n0,
                    (uint)(nLen * sizeof(float)));
            }
        }
        else
        {
            // B is stored as [N, K]; pack B^T[k0..k0+kLen, n0..n0+nLen] into [kLen, nLen]
            for (int k = 0; k < kLen; k++)
            {
                float* dstRow = dst + (long)k * nLen;
                int colSrc = k0 + k;
                for (int n = 0; n < nLen; n++)
                {
                    dstRow[n] = srcB[(long)(n0 + n) * ldbSrc + colSrc];
                }
            }
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

                for (; n + 8 <= N; n += 8)
                {
                    MicroKernelGemmAvx2_8x8(
                        A + (long)m * lda,
                        B + n,
                        C + (long)m * ldc + n,
                        lda, ldb, ldc, K);
                }

                for (; n + 4 <= N; n += 4)
                {
                    MicroKernelGemmVec128_4x4(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, K);
                    MicroKernelGemmVec128_4x4(A + (long)(m + 4) * lda, B + n, C + (long)(m + 4) * ldc + n, lda, ldb, ldc, K);
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

                for (; n + 8 <= N; n += 8)
                {
                    MicroKernelGemmAvx2_4x8(
                        A + (long)m * lda,
                        B + n,
                        C + (long)m * ldc + n,
                        lda, ldb, ldc, K);
                }

                for (; n + 4 <= N; n += 4)
                {
                    MicroKernelGemmVec128_4x4(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, K);
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

                for (; n + 4 <= N; n += 4)
                {
                    MicroKernelGemmVec128_4x4(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, K);
                    MicroKernelGemmVec128_4x4(A + (long)(m + 4) * lda, B + n, C + (long)(m + 4) * ldc + n, lda, ldb, ldc, K);
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

                for (; n + 4 <= N; n += 4)
                {
                    MicroKernelGemmVec128_4x4(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, K);
                }

                for (; n < N; n++)
                {
                    ScalarColumn(A + (long)m * lda, B + n, C + (long)m * ldc + n, lda, ldb, ldc, 4, K);
                }
            }
        }
        else if (AdvSimd.IsSupported || Vector128.IsHardwareAccelerated)
        {
            for (; m + 4 <= M; m += 4)
            {
                int n = 0;
                for (; n + 4 <= N; n += 4)
                {
                    MicroKernelGemmAdvSimd_4x4(
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

        // Vectorized remainder rows
        for (; m < M; m++)
        {
            float* aRow = A + (long)m * lda;
            float* cRow = C + (long)m * ldc;
            int n = 0;

            if (Vector512.IsHardwareAccelerated)
            {
                for (; n + 16 <= N; n += 16)
                {
                    var acc = Vector512.Load(cRow + n);
                    for (int k = 0; k < K; k++)
                    {
                        var va = Vector512.Create(aRow[k]);
                        var vb = Vector512.Load(B + (long)k * ldb + n);
                        acc = Avx512F.IsSupported ? Avx512F.FusedMultiplyAdd(va, vb, acc) : acc + va * vb;
                    }
                    Vector512.Store(acc, cRow + n);
                }
            }

            if (Vector256.IsHardwareAccelerated)
            {
                for (; n + 8 <= N; n += 8)
                {
                    var acc = Vector256.Load(cRow + n);
                    for (int k = 0; k < K; k++)
                    {
                        var va = Vector256.Create(aRow[k]);
                        var vb = Vector256.Load(B + (long)k * ldb + n);
                        acc = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, acc) : acc + va * vb;
                    }
                    Vector256.Store(acc, cRow + n);
                }
            }

            if (Vector128.IsHardwareAccelerated)
            {
                for (; n + 4 <= N; n += 4)
                {
                    var acc = Vector128.Load(cRow + n);
                    for (int k = 0; k < K; k++)
                    {
                        var va = Vector128.Create(aRow[k]);
                        var vb = Vector128.Load(B + (long)k * ldb + n);
                        acc = acc + va * vb;
                    }
                    Vector128.Store(acc, cRow + n);
                }
            }

            for (; n < N; n++)
            {
                float sum = 0f;
                for (int k = 0; k < K; k++)
                {
                    sum += aRow[k] * B[(long)k * ldb + n];
                }
                cRow[n] += sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MicroKernelGemmAdvSimd_4x4(
        float* A, float* B, float* C,
        int lda, int ldb, int ldc,
        int K)
    {
        if (AdvSimd.IsSupported)
        {
            Vector128<float> c0 = Vector128.Load(C + 0 * ldc);
            Vector128<float> c1 = Vector128.Load(C + 1 * ldc);
            Vector128<float> c2 = Vector128.Load(C + 2 * ldc);
            Vector128<float> c3 = Vector128.Load(C + 3 * ldc);

            for (int k = 0; k < K; k++)
            {
                Vector128<float> bRow = Vector128.Load(B + (long)k * ldb);
                c0 = AdvSimd.FusedMultiplyAdd(c0, Vector128.Create(*(A + 0 * lda + k)), bRow);
                c1 = AdvSimd.FusedMultiplyAdd(c1, Vector128.Create(*(A + 1 * lda + k)), bRow);
                c2 = AdvSimd.FusedMultiplyAdd(c2, Vector128.Create(*(A + 2 * lda + k)), bRow);
                c3 = AdvSimd.FusedMultiplyAdd(c3, Vector128.Create(*(A + 3 * lda + k)), bRow);
            }

            Vector128.Store(c0, C + 0 * ldc);
            Vector128.Store(c1, C + 1 * ldc);
            Vector128.Store(c2, C + 2 * ldc);
            Vector128.Store(c3, C + 3 * ldc);
        }
        else
        {
            MicroKernelGemmVec128_4x4(A, B, C, lda, ldb, ldc, K);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MicroKernelGemmVec128_4x4(
        float* A, float* B, float* C,
        int lda, int ldb, int ldc,
        int K)
    {
        Vector128<float> c0 = Vector128.Load(C + 0 * ldc);
        Vector128<float> c1 = Vector128.Load(C + 1 * ldc);
        Vector128<float> c2 = Vector128.Load(C + 2 * ldc);
        Vector128<float> c3 = Vector128.Load(C + 3 * ldc);

        for (int k = 0; k < K; k++)
        {
            Vector128<float> bRow = Vector128.Load(B + (long)k * ldb);
            if (Fma.IsSupported)
            {
                c0 = Fma.MultiplyAdd(Vector128.Create(*(A + 0 * lda + k)), bRow, c0);
                c1 = Fma.MultiplyAdd(Vector128.Create(*(A + 1 * lda + k)), bRow, c1);
                c2 = Fma.MultiplyAdd(Vector128.Create(*(A + 2 * lda + k)), bRow, c2);
                c3 = Fma.MultiplyAdd(Vector128.Create(*(A + 3 * lda + k)), bRow, c3);
            }
            else
            {
                c0 = Vector128.Add(c0, Vector128.Multiply(Vector128.Create(*(A + 0 * lda + k)), bRow));
                c1 = Vector128.Add(c1, Vector128.Multiply(Vector128.Create(*(A + 1 * lda + k)), bRow));
                c2 = Vector128.Add(c2, Vector128.Multiply(Vector128.Create(*(A + 2 * lda + k)), bRow));
                c3 = Vector128.Add(c3, Vector128.Multiply(Vector128.Create(*(A + 3 * lda + k)), bRow));
            }
        }

        Vector128.Store(c0, C + 0 * ldc);
        Vector128.Store(c1, C + 1 * ldc);
        Vector128.Store(c2, C + 2 * ldc);
        Vector128.Store(c3, C + 3 * ldc);
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
