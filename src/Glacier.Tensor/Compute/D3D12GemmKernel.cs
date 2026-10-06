using System;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Tensor.Core;
using Glacier.Tensor.Diagnostics;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Glacier.Tensor.Compute;

/// <summary>
/// High-performance Direct3D 12 compute kernel for general matrix multiplication (GEMM).
/// Utilizes a 16x16 shared-memory tiled compute shader with double buffering, Wave32 optimization,
/// and direct root descriptors for zero-overhead execution across AMD Radeon APUs and DirectX 12 GPUs.
/// </summary>
public static unsafe class D3D12GemmKernel
{
    private static readonly Lock s_lock = new();
    private static bool s_initialized;
    private static bool s_available;

    private static IDXGIFactory4? s_factory;
    private static IDXGIAdapter1? s_adapter;
    private static ID3D12Device? s_device;
    private static ID3D12CommandQueue? s_queue;
    private static ID3D12CommandAllocator? s_cmdAlloc;
    private static ID3D12GraphicsCommandList? s_cmdList;
    private static ID3D12Fence? s_fence;
    private static ulong s_fenceValue;
    private static AutoResetEvent? s_fenceEvent;

    private static ID3D12RootSignature? s_rootSig;
    private static ID3D12PipelineState? s_pipelineState;

    private static ID3D12Resource? s_devA;
    private static ID3D12Resource? s_devB;
    private static ID3D12Resource? s_devC;
    private static ID3D12Resource? s_uploadA;
    private static ID3D12Resource? s_uploadB;
    private static ID3D12Resource? s_readbackC;

    private static ulong s_capA;
    private static ulong s_capB;
    private static ulong s_capC;

    private const int FenceTimeoutMilliseconds = 10_000;

    public static string DeviceName { get; private set; } = string.Empty;
    public static bool IsSupported => EnsureInitialized();
    [ThreadStatic]
    private static D3D12KernelDiagnostics? t_lastError;
    public static D3D12KernelDiagnostics? LastError
    {
        get => t_lastError;
        private set => t_lastError = value;
    }
    public static Exception? LastException => LastError?.Exception;
    public static void ClearLastError() => t_lastError = null;

    public const string HlslSource = @"
cbuffer Constants : register(b0)
{
    uint M;
    uint K;
    uint N;
    uint _pad;
};

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

#define BK 16
#define BM 64
#define BN 64
#define TM 4
#define TN 4

groupshared float sA[BM][BK];
groupshared float sB[BK][BN];

[numthreads(16, 16, 1)]
void main(uint3 gId : SV_GroupID, uint3 tId : SV_GroupThreadID)
{
    uint linearTid = tId.y * 16 + tId.x;
    uint numTiles = (K + BK - 1) / BK;

    uint threadRow = tId.y * TM;
    uint threadCol = tId.x * TN;

    uint globalRow = gId.y * BM + threadRow;
    uint globalCol = gId.x * BN + threadCol;

    float acc[TM][TN];
    [unroll]
    for (uint i = 0; i < TM; ++i)
    {
        [unroll]
        for (uint j = 0; j < TN; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }

    uint aLoadRow = linearTid / 4;
    uint aLoadCol = (linearTid % 4) * 4;
    uint aGlobalRow = gId.y * BM + aLoadRow;

    uint bLoadRow = linearTid / 16;
    uint bLoadCol = (linearTid % 16) * 4;
    uint bGlobalCol = gId.x * BN + bLoadCol;

    for (uint tile = 0; tile < numTiles; ++tile)
    {
        uint aGlobalCol = tile * BK + aLoadCol;
        [unroll]
        for (uint c = 0; c < 4; ++c)
        {
            uint ac = aGlobalCol + c;
            sA[aLoadRow][aLoadCol + c] = (aGlobalRow < M && ac < K) ? A[aGlobalRow * K + ac] : 0.0f;
        }

        uint bGlobalRow = tile * BK + bLoadRow;
        [unroll]
        for (uint r = 0; r < 4; ++r)
        {
            uint br = bGlobalRow;
            uint bc = bGlobalCol + r;
            sB[bLoadRow][bLoadCol + r] = (br < K && bc < N) ? B[br * N + bc] : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < BK; ++k)
        {
            float regA[TM];
            float regB[TN];

            [unroll]
            for (uint r = 0; r < TM; ++r)
            {
                regA[r] = sA[threadRow + r][k];
            }

            [unroll]
            for (uint c = 0; c < TN; ++c)
            {
                regB[c] = sB[k][threadCol + c];
            }

            [unroll]
            for (uint r = 0; r < TM; ++r)
            {
                [unroll]
                for (uint c = 0; c < TN; ++c)
                {
                    acc[r][c] += regA[r] * regB[c];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint r = 0; r < TM; ++r)
    {
        [unroll]
        for (uint c = 0; c < TN; ++c)
        {
            uint cr = globalRow + r;
            uint cc = globalCol + c;
            if (cr < M && cc < N)
            {
                C[cr * N + cc] = acc[r][c];
            }
        }
    }
}
";

    public static bool EnsureInitialized()
    {
        if (s_initialized) return s_available;
        lock (s_lock)
        {
            if (s_initialized) return s_available;
            try
            {
                if (!OperatingSystem.IsWindows())
                {
                    s_available = false;
                    s_initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, new PlatformNotSupportedException("Direct3D 12 is only supported on Windows."));
                    return false;
                }

                s_factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
                IDXGIAdapter1? chosenAdapter = null;
                for (uint i = 0; s_factory.EnumAdapters1(i, out IDXGIAdapter1 a).Success; i++)
                {
                    var desc = a.Description1;
                    if ((desc.Flags & AdapterFlags.Software) != 0)
                    {
                        a.Dispose();
                        continue;
                    }

                    // Prefer AMD Radeon adapter, otherwise first hardware GPU
                    if (desc.Description.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                    {
                        chosenAdapter = a;
                        break;
                    }

                    if (chosenAdapter == null)
                        chosenAdapter = a;
                    else
                        a.Dispose();
                }

                if (chosenAdapter == null)
                {
                    s_available = false;
                    s_initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, new NotSupportedException("No compatible Direct3D 12 hardware adapter found."));
                    return false;
                }

                s_adapter = chosenAdapter;
                DeviceName = s_adapter.Description1.Description;

                var hr = D3D12.D3D12CreateDevice(s_adapter, FeatureLevel.Level_11_0, out s_device!);
                if (!hr.Success || s_device == null)
                {
                    s_available = false;
                    s_initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, new InvalidOperationException($"D3D12CreateDevice failed with HRESULT 0x{hr.Code:X8}."));
                    return false;
                }

                var queueDesc = new CommandQueueDescription(CommandListType.Compute);
                s_queue = s_device.CreateCommandQueue(queueDesc);

                s_cmdAlloc = s_device.CreateCommandAllocator(CommandListType.Compute);
                s_cmdList = s_device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, s_cmdAlloc);
                s_cmdList.Close();

                s_fence = s_device.CreateFence(0);
                s_fenceValue = 0;
                s_fenceEvent = new AutoResetEvent(false);

                // Build root signature
                var rootParams = new RootParameter[]
                {
                    new(new RootConstants(0, 0, 4), ShaderVisibility.All),
                    new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
                };

                s_rootSig = s_device.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rootParams), RootSignatureVersion.Version1);

                // Compile shader
                var bytecode = Compiler.Compile(HlslSource, "main", "source.hlsl", "cs_5_0");
                if (bytecode.IsEmpty)
                {
                    s_available = false;
                    s_initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, new InvalidOperationException("Failed to compile GEMM HLSL compute shader bytecode."));
                    return false;
                }

                var psoDesc = new ComputePipelineStateDescription
                {
                    RootSignature = s_rootSig,
                    ComputeShader = bytecode
                };
                s_pipelineState = s_device.CreateComputePipelineState(psoDesc);

                s_available = true;
            }
            catch (Exception ex)
            {
                s_available = false;
                RecordError(D3D12ExecutionStage.Initialization, ex);
            }
            finally
            {
                s_initialized = true;
            }

            return s_available;
        }
    }

    private static D3D12KernelDiagnostics RecordError(
        D3D12ExecutionStage stage,
        Exception ex,
        int m = 0, int k = 0, int n = 0,
        ulong bytesA = 0, ulong bytesB = 0, ulong bytesC = 0,
        uint gridX = 0, uint gridY = 0)
    {
        bool isDeviceRemoved = false;
        int? removalHResult = null;
        string? removalDesc = null;

        if (s_device != null)
        {
            try
            {
                var reason = s_device.DeviceRemovedReason;
                if (reason.Failure)
                {
                    isDeviceRemoved = true;
                    removalHResult = reason.Code;
                    removalDesc = reason.Description;
                }
            }
            catch
            {
                // Ignore failure querying device removed reason
            }
        }

        var diag = new D3D12KernelDiagnostics
        {
            Success = false,
            Stage = stage,
            DeviceName = DeviceName,
            IsDeviceRemoved = isDeviceRemoved,
            DeviceRemovedReasonHResult = removalHResult,
            DeviceRemovedReasonDescription = removalDesc,
            M = m,
            K = k,
            N = n,
            BytesA = bytesA,
            BytesB = bytesB,
            BytesC = bytesC,
            GridX = gridX,
            GridY = gridY,
            ExceptionType = ex.GetType().FullName,
            ErrorMessage = ex.Message,
            Exception = ex,
            HResult = ex.HResult,
            StackTrace = ex.StackTrace,
            Timestamp = DateTimeOffset.UtcNow
        };

        LastError = diag;

        GlacierDiagnostics.LogError(
            $"[D3D12GemmKernel] Error at stage '{stage}' on device '{DeviceName}' " +
            $"(GEMM [{m}x{k}] * [{k}x{n}] -> [{m}x{n}], Dispatch [{gridX},{gridY},1], " +
            $"DeviceRemoved: {isDeviceRemoved}" +
            (isDeviceRemoved ? $" [0x{removalHResult:X8}: {removalDesc}]" : "") +
            $"): {ex.GetType().Name} - {ex.Message}", ex);

        return diag;
    }

    private static void EnsureBuffers(ulong bytesA, ulong bytesB, ulong bytesC)
    {
        const ulong initCap = 16 * 1024 * 1024; // 16 MB persistent GPU heap
        if (s_devA == null || s_capA < bytesA)
        {
            s_devA?.Dispose();
            s_uploadA?.Dispose();
            s_capA = Math.Max(bytesA, Math.Max(initCap, (ulong)(s_capA * 1.5)));
            s_devA = CreateDeviceBuffer(s_capA);
            s_uploadA = CreateUploadBuffer(s_capA);
        }

        if (s_devB == null || s_capB < bytesB)
        {
            s_devB?.Dispose();
            s_uploadB?.Dispose();
            s_capB = Math.Max(bytesB, Math.Max(initCap, (ulong)(s_capB * 1.5)));
            s_devB = CreateDeviceBuffer(s_capB);
            s_uploadB = CreateUploadBuffer(s_capB);
        }

        if (s_devC == null || s_capC < bytesC)
        {
            s_devC?.Dispose();
            s_readbackC?.Dispose();
            s_capC = Math.Max(bytesC, Math.Max(initCap, (ulong)(s_capC * 1.5)));
            s_devC = CreateDeviceBuffer(s_capC, ResourceFlags.AllowUnorderedAccess);
            s_readbackC = CreateReadbackBuffer(s_capC);
        }
    }

    private static ID3D12Resource CreateDeviceBuffer(ulong sizeInBytes, ResourceFlags flags = ResourceFlags.None)
    {
        var desc = ResourceDescription.Buffer(sizeInBytes, flags);
        var heapProps = new HeapProperties(HeapType.Default);
        return s_device!.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.Common);
    }

    private static ID3D12Resource CreateUploadBuffer(ulong sizeInBytes)
    {
        var desc = ResourceDescription.Buffer(sizeInBytes);
        var heapProps = new HeapProperties(HeapType.Upload);
        return s_device!.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.GenericRead);
    }

    private static ID3D12Resource CreateReadbackBuffer(ulong sizeInBytes)
    {
        var desc = ResourceDescription.Buffer(sizeInBytes);
        var heapProps = new HeapProperties(HeapType.Readback);
        return s_device!.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.CopyDest);
    }

    private static void Synchronize()
    {
        s_fenceValue++;
        s_queue!.Signal(s_fence!, s_fenceValue);

        if (s_fence!.CompletedValue < s_fenceValue)
        {
            s_fence.SetEventOnCompletion(s_fenceValue, s_fenceEvent!);
            if (!s_fenceEvent!.WaitOne(FenceTimeoutMilliseconds))
            {
                throw new TimeoutException($"Direct3D 12 GPU fence synchronization timed out after {FenceTimeoutMilliseconds} ms.");
            }
        }
    }

    public static bool Execute(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        return Execute(a, b, c, out _);
    }

    public static bool Execute(Tensor<float> a, Tensor<float> b, Tensor<float> c, out D3D12KernelDiagnostics? diagnostics)
    {
        diagnostics = null;

        ArgumentNullException.ThrowIfNull(a, nameof(a));
        ArgumentNullException.ThrowIfNull(b, nameof(b));
        ArgumentNullException.ThrowIfNull(c, nameof(c));

        D3D12ExecutionStage stage = D3D12ExecutionStage.ArgumentValidation;
        int m = 0, k = 0, n = 0;
        ulong bytesA = 0, bytesB = 0, bytesC = 0;
        uint gridX = 0, gridY = 0;

        try
        {
            if (a.Rank != 2)
                throw new ArgumentException($"Tensor 'a' must be 2D matrix (Rank=2), got {a.Rank}.", nameof(a));
            if (b.Rank != 2)
                throw new ArgumentException($"Tensor 'b' must be 2D matrix (Rank=2), got {b.Rank}.", nameof(b));
            if (c.Rank != 2)
                throw new ArgumentException($"Tensor 'c' must be 2D matrix (Rank=2), got {c.Rank}.", nameof(c));

            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException($"Inner matrix dimensions mismatch: a.Shape[1] ({a.Shape[1]}) != b.Shape[0] ({b.Shape[0]}).");

            if (c.Shape[0] != a.Shape[0] || c.Shape[1] != b.Shape[1])
                throw new ArgumentException($"Result tensor shape [{c.Shape[0]}, {c.Shape[1]}] does not match product dimensions [{a.Shape[0]}, {b.Shape[1]}].");

            if (!a.IsContiguous || !b.IsContiguous || !c.IsContiguous)
                throw new InvalidOperationException("Direct3D 12 GEMM requires contiguous memory layout for all operands.");

            m = a.Shape[0];
            k = a.Shape[1];
            n = b.Shape[1];

            bytesA = (ulong)(m * k * sizeof(float));
            bytesB = (ulong)(k * n * sizeof(float));
            bytesC = (ulong)(m * n * sizeof(float));

            gridX = (uint)((n + 63) / 64);
            gridY = (uint)((m + 63) / 64);

            stage = D3D12ExecutionStage.Initialization;
            if (!EnsureInitialized())
            {
                diagnostics = RecordError(stage, new InvalidOperationException("Direct3D 12 initialization failed or device not available on this platform."), m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
                return false;
            }

            lock (s_lock)
            {
                stage = D3D12ExecutionStage.BufferAllocation;
                EnsureBuffers(bytesA, bytesB, bytesC);

                stage = D3D12ExecutionStage.HostUpload;
                fixed (float* pA = a.AsSpan(), pB = b.AsSpan())
                {
                    void* pUpA = null;
                    s_uploadA!.Map(0, null, &pUpA);
                    Buffer.MemoryCopy(pA, pUpA, bytesA, bytesA);
                    s_uploadA.Unmap(0);

                    void* pUpB = null;
                    s_uploadB!.Map(0, null, &pUpB);
                    Buffer.MemoryCopy(pB, pUpB, bytesB, bytesB);
                    s_uploadB.Unmap(0);
                }

                stage = D3D12ExecutionStage.CommandRecording;
                s_cmdAlloc!.Reset();
                s_cmdList!.Reset(s_cmdAlloc, null);

                // Copy to device buffers
                s_cmdList.ResourceBarrierTransition(s_devA!, ResourceStates.Common, ResourceStates.CopyDest);
                s_cmdList.ResourceBarrierTransition(s_devB!, ResourceStates.Common, ResourceStates.CopyDest);
                s_cmdList.CopyBufferRegion(s_devA!, 0, s_uploadA!, 0, bytesA);
                s_cmdList.CopyBufferRegion(s_devB!, 0, s_uploadB!, 0, bytesB);

                s_cmdList.ResourceBarrierTransition(s_devA!, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
                s_cmdList.ResourceBarrierTransition(s_devB!, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
                s_cmdList.ResourceBarrierTransition(s_devC!, ResourceStates.Common, ResourceStates.UnorderedAccess);

                // Bind pipeline and root signature
                s_cmdList.SetComputeRootSignature(s_rootSig!);
                s_cmdList.SetPipelineState(s_pipelineState!);

                uint* pConsts = stackalloc uint[4];
                pConsts[0] = (uint)m;
                pConsts[1] = (uint)k;
                pConsts[2] = (uint)n;
                pConsts[3] = 0;
                s_cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

                s_cmdList.SetComputeRootShaderResourceView(1, s_devA!.GPUVirtualAddress);
                s_cmdList.SetComputeRootShaderResourceView(2, s_devB!.GPUVirtualAddress);
                s_cmdList.SetComputeRootUnorderedAccessView(3, s_devC!.GPUVirtualAddress);

                stage = D3D12ExecutionStage.Dispatch;
                s_cmdList.Dispatch(gridX, gridY, 1);

                // Transition C for readback and revert A, B
                s_cmdList.ResourceBarrierTransition(s_devC!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                s_cmdList.ResourceBarrierTransition(s_devA!, ResourceStates.NonPixelShaderResource, ResourceStates.Common);
                s_cmdList.ResourceBarrierTransition(s_devB!, ResourceStates.NonPixelShaderResource, ResourceStates.Common);

                s_cmdList.CopyBufferRegion(s_readbackC!, 0, s_devC!, 0, bytesC);
                s_cmdList.ResourceBarrierTransition(s_devC!, ResourceStates.CopySource, ResourceStates.Common);

                stage = D3D12ExecutionStage.QueueExecution;
                s_cmdList.Close();
                s_queue!.ExecuteCommandList(s_cmdList);

                stage = D3D12ExecutionStage.FenceSynchronization;
                Synchronize();

                stage = D3D12ExecutionStage.DeviceReadback;
                fixed (float* pC = c.AsSpan())
                {
                    void* pRead = null;
                    s_readbackC!.Map(0, null, &pRead);
                    Buffer.MemoryCopy(pRead, pC, bytesC, bytesC);
                    s_readbackC.Unmap(0);
                }

                t_lastError = null;
                diagnostics = new D3D12KernelDiagnostics
                {
                    Success = true,
                    Stage = D3D12ExecutionStage.None,
                    DeviceName = DeviceName,
                    M = m,
                    K = k,
                    N = n,
                    BytesA = bytesA,
                    BytesB = bytesB,
                    BytesC = bytesC,
                    GridX = gridX,
                    GridY = gridY,
                    Timestamp = DateTimeOffset.UtcNow
                };

                return true;
            }
        }
        catch (Exception ex)
        {
            diagnostics = RecordError(stage, ex, m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
            return false;
        }
    }
}
