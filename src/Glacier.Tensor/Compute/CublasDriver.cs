using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Optional dynamic cuBLAS driver wrapper (<c>cublas64_13.dll</c> / <c>cublas64_12.dll</c> / <c>libcublas.so</c>).
/// Loaded dynamically with zero hard CUDA Toolkit dependency; callers fall back seamlessly to embedded PTX when absent.
/// </summary>
internal static unsafe class CublasDriver
{
    private const int CUBLAS_OP_N = 0;
    private const int CUBLAS_OP_T = 1;
    private const int CUBLAS_STATUS_SUCCESS = 0;

    private static readonly object s_loadLock = new();
    private static bool s_probed;
    private static bool s_available;
    private static IntPtr s_library;

    private static delegate* unmanaged[Cdecl]<IntPtr*, int> s_cublasCreate;
    private static delegate* unmanaged[Cdecl]<IntPtr, int> s_cublasDestroy;
    private static delegate* unmanaged[Cdecl]<IntPtr, IntPtr, int> s_cublasSetStream;
    private static delegate* unmanaged[Cdecl]<
        IntPtr, int, int,
        int, int, int,
        float*, IntPtr, int,
        IntPtr, int,
        float*, IntPtr, int, int> s_cublasSgemm;

    private static readonly ConcurrentQueue<IntPtr> s_handlePool = new();

    static CublasDriver()
    {
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => ClearHandles();
    }

    public static bool IsAvailable => EnsureLoaded();

    private static bool EnsureLoaded()
    {
        if (s_probed) return s_available;
        lock (s_loadLock)
        {
            if (s_probed) return s_available;
            try
            {
                s_available = LoadCore();
            }
            catch
            {
                s_available = false;
            }
            finally
            {
                s_probed = true;
            }
            return s_available;
        }
    }

    private static bool LoadCore()
    {
        string[] candidates = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? ["cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll"]
            : ["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"];

        IntPtr lib = IntPtr.Zero;
        foreach (string name in candidates)
        {
            if (NativeLibrary.TryLoad(name, out lib) && lib != IntPtr.Zero)
                break;
        }

        if (lib == IntPtr.Zero && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            string? cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath))
            {
                foreach (string name in candidates)
                {
                    string full = System.IO.Path.Combine(cudaPath, "bin", name);
                    if (System.IO.File.Exists(full) && NativeLibrary.TryLoad(full, out lib) && lib != IntPtr.Zero)
                        break;
                    string fullX64 = System.IO.Path.Combine(cudaPath, "bin", "x64", name);
                    if (System.IO.File.Exists(fullX64) && NativeLibrary.TryLoad(fullX64, out lib) && lib != IntPtr.Zero)
                        break;
                }
            }
        }

        if (lib == IntPtr.Zero) return false;

        if (!NativeLibrary.TryGetExport(lib, "cublasCreate_v2", out IntPtr pCreate) ||
            !NativeLibrary.TryGetExport(lib, "cublasDestroy_v2", out IntPtr pDestroy) ||
            !NativeLibrary.TryGetExport(lib, "cublasSetStream_v2", out IntPtr pSetStream) ||
            !NativeLibrary.TryGetExport(lib, "cublasSgemm_v2", out IntPtr pSgemm))
        {
            NativeLibrary.Free(lib);
            return false;
        }

        s_library = lib;
        s_cublasCreate = (delegate* unmanaged[Cdecl]<IntPtr*, int>)pCreate;
        s_cublasDestroy = (delegate* unmanaged[Cdecl]<IntPtr, int>)pDestroy;
        s_cublasSetStream = (delegate* unmanaged[Cdecl]<IntPtr, IntPtr, int>)pSetStream;
        s_cublasSgemm = (delegate* unmanaged[Cdecl]<IntPtr, int, int, int, int, int, float*, IntPtr, int, IntPtr, int, float*, IntPtr, int, int>)pSgemm;
        return true;
    }

    public static bool TrySgemm(
        IntPtr stream,
        bool transposeA,
        bool transposeB,
        int m,
        int n,
        int k,
        IntPtr devA,
        IntPtr devB,
        IntPtr devC)
    {
        if (!EnsureLoaded() || devA == IntPtr.Zero || devB == IntPtr.Zero || devC == IntPtr.Zero)
            return false;

        if (!s_handlePool.TryDequeue(out IntPtr handle))
        {
            IntPtr created = IntPtr.Zero;
            if (s_cublasCreate(&created) != CUBLAS_STATUS_SUCCESS || created == IntPtr.Zero)
                return false;
            handle = created;
        }

        bool healthy = true;
        try
        {
            if (s_cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
            {
                healthy = false;
                return false;
            }

            // Row-major C[M, N] = op(A)[M, K] * op(B)[K, N]
            // Equivalent column-major: C^T[N, M] = op(B)^T[N, K] * op(A)^T[K, M]
            float alpha = 1.0f;
            float beta = 0.0f;
            int opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
            int opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
            int ldb = transposeB ? k : n;
            int lda = transposeA ? m : k;
            int ldc = n;

            int status = s_cublasSgemm(
                handle,
                opB, opA,
                n, m, k,
                &alpha,
                devB, ldb,
                devA, lda,
                &beta,
                devC, ldc);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                healthy = false;
                return false;
            }
            return true;
        }
        catch
        {
            healthy = false;
            return false;
        }
        finally
        {
            if (healthy)
            {
                s_handlePool.Enqueue(handle);
            }
            else if (handle != IntPtr.Zero && s_cublasDestroy != null)
            {
                try { s_cublasDestroy(handle); } catch { }
            }
        }
    }

    public static void ClearHandles()
    {
        if (!s_available || s_cublasDestroy == null) return;
        while (s_handlePool.TryDequeue(out IntPtr handle))
        {
            if (handle != IntPtr.Zero)
            {
                try { s_cublasDestroy(handle); } catch { }
            }
        }
    }
}
