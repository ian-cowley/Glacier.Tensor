using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security.Cryptography;
using System.Text;
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
/// Utilizes a 64x64 register-tiled (4x4 per thread) double-buffered shared-memory compute shader,
/// cached compiled shader bytecode (.cso), and 2-slot command execution for zero-overhead execution
/// across AMD Radeon APUs and DirectX 12 GPUs.
/// </summary>
public static unsafe partial class D3D12GemmKernel
{
    private sealed class ExecutionSlot
    {
        public ID3D12CommandAllocator? Allocator;
        public ID3D12GraphicsCommandList? CommandList;
        public ulong FenceValue;
    }

    private static readonly Lock s_shaderLock = new();
    private static byte[]? s_cachedShaderBytecode;

    private static readonly D3D12KernelInstance s_default = new(requiredVendorId: null, updateThreadDiagnostics: true);
    private static readonly D3D12KernelInstance s_amd = new(requiredVendorId: 0x1002u, updateThreadDiagnostics: false);

    private const int FenceTimeoutMilliseconds = 10_000;

    static D3D12GemmKernel()
    {
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => CleanupProcessExit();
    }

    private static void CleanupProcessExit()
    {
        s_default.Cleanup();
        s_amd.Cleanup();
    }

    public static string DeviceName => s_default.DeviceName;
    public static bool IsSupported => s_default.EnsureInitialized();
    public static ID3D12Device? Device
    {
        get
        {
            s_default.EnsureInitialized();
            return s_default.Device;
        }
    }

    public static bool IsAmdSupported =>
        OperatingSystem.IsWindows() && s_amd.EnsureInitialized() && s_amd.VendorId == 0x1002u;
    public static string AmdDeviceName => s_amd.DeviceName;
    public static uint AmdVendorId => s_amd.VendorId;

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

groupshared float sA[2][BM][BK];
groupshared float sB[2][BK][BN];

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

    if (numTiles > 0)
    {
        uint aGlobalCol0 = aLoadCol;
        [unroll]
        for (uint c0 = 0; c0 < 4; ++c0)
        {
            uint ac = aGlobalCol0 + c0;
            sA[0][aLoadRow][aLoadCol + c0] = (aGlobalRow < M && ac < K) ? A[aGlobalRow * K + ac] : 0.0f;
        }

        uint bGlobalRow0 = bLoadRow;
        [unroll]
        for (uint r0 = 0; r0 < 4; ++r0)
        {
            uint bc = bGlobalCol + r0;
            sB[0][bLoadRow][bLoadCol + r0] = (bGlobalRow0 < K && bc < N) ? B[bGlobalRow0 * N + bc] : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        uint bufIdx = 0;
        for (uint tile = 0; tile < numTiles; ++tile)
        {
            float nextA[4];
            float nextB[4];
            uint nextTile = tile + 1;
            if (nextTile < numTiles)
            {
                uint aGlobalColNext = nextTile * BK + aLoadCol;
                [unroll]
                for (uint c1 = 0; c1 < 4; ++c1)
                {
                    uint ac = aGlobalColNext + c1;
                    nextA[c1] = (aGlobalRow < M && ac < K) ? A[aGlobalRow * K + ac] : 0.0f;
                }

                uint bGlobalRowNext = nextTile * BK + bLoadRow;
                [unroll]
                for (uint r1 = 0; r1 < 4; ++r1)
                {
                    uint bc = bGlobalCol + r1;
                    nextB[r1] = (bGlobalRowNext < K && bc < N) ? B[bGlobalRowNext * N + bc] : 0.0f;
                }
            }

            [unroll]
            for (uint k = 0; k < BK; ++k)
            {
                float regA[TM];
                float regB[TN];

                [unroll]
                for (uint r = 0; r < TM; ++r)
                {
                    regA[r] = sA[bufIdx][threadRow + r][k];
                }

                [unroll]
                for (uint c = 0; c < TN; ++c)
                {
                    regB[c] = sB[bufIdx][k][threadCol + c];
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

            if (nextTile < numTiles)
            {
                uint nextBuf = bufIdx ^ 1;
                [unroll]
                for (uint c2 = 0; c2 < 4; ++c2)
                {
                    sA[nextBuf][aLoadRow][aLoadCol + c2] = nextA[c2];
                }

                [unroll]
                for (uint r2 = 0; r2 < 4; ++r2)
                {
                    sB[nextBuf][bLoadRow][bLoadCol + r2] = nextB[r2];
                }

                GroupMemoryBarrierWithGroupSync();
                bufIdx = nextBuf;
            }
        }
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

    public static bool EnsureInitialized() => s_default.EnsureInitialized();

    [SupportedOSPlatform("windows")]
    private static byte[] LoadOrCompileShaderBytecode()
    {
        lock (s_shaderLock)
        {
            if (s_cachedShaderBytecode is { Length: > 0 } memCached)
            {
                return memCached;
            }

            string? cachePath = null;
            try
            {
                byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes($"cs_5_0:main:{HlslSource}"));
                string hashHex = Convert.ToHexString(hash.AsSpan(0, 16));
                string baseDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                if (string.IsNullOrWhiteSpace(baseDir))
                {
                    baseDir = Path.GetTempPath();
                }

                string cacheDir = Path.Combine(baseDir, "Glacier", "ShaderCache");
                Directory.CreateDirectory(cacheDir);
                cachePath = Path.Combine(cacheDir, $"d3d12_gemm_{hashHex}.cso");
                if (File.Exists(cachePath))
                {
                    byte[] diskBytes = File.ReadAllBytes(cachePath);
                    if (diskBytes.Length > 0)
                    {
                        s_cachedShaderBytecode = diskBytes;
                        return diskBytes;
                    }
                }
            }
            catch
            {
                cachePath = null;
            }

            var compiled = Compiler.Compile(HlslSource, "main", "source.hlsl", "cs_5_0");
            if (compiled.IsEmpty)
            {
                return Array.Empty<byte>();
            }

            byte[] result = compiled.ToArray();
            s_cachedShaderBytecode = result;

            if (!string.IsNullOrEmpty(cachePath))
            {
                string tmpPath = $"{cachePath}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";
                try
                {
                    File.WriteAllBytes(tmpPath, result);
                    File.Move(tmpPath, cachePath, overwrite: true);
                }
                catch (Exception cleanupEx) when (cleanupEx is IOException or UnauthorizedAccessException)
                {
                    try
                    {
                        if (File.Exists(tmpPath)) File.Delete(tmpPath);
                    }
                    catch (Exception deleteEx) when (deleteEx is IOException or UnauthorizedAccessException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"[D3D12GemmKernel] Failed to delete temp shader cache '{tmpPath}': {deleteEx.Message}");
                    }
                    System.Diagnostics.Trace.TraceWarning($"[D3D12GemmKernel] Failed to persist shader cache '{cachePath}': {cleanupEx.Message}");
                }
            }

            return result;
        }
    }

    public static bool Execute(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        return s_default.Execute(a, b, c, out _);
    }

    public static bool Execute(Tensor<float> a, Tensor<float> b, Tensor<float> c, out D3D12KernelDiagnostics? diagnostics)
    {
        return s_default.Execute(a, b, c, out diagnostics);
    }

    public static bool ExecuteAmd(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        return s_amd.Execute(a, b, c, out _);
    }

    public static bool ExecuteAmd(Tensor<float> a, Tensor<float> b, Tensor<float> c, out D3D12KernelDiagnostics? diagnostics)
    {
        return s_amd.Execute(a, b, c, out diagnostics);
    }

    /// <summary>
    /// Pre-loads operands A and B into GPU device-resident VRAM buffers across PCIe.
    /// </summary>
    public static void UploadBuffers(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int m, int k, int n)
    {
        s_default.UploadBuffers(a, b, m, k, n);
    }

    /// <summary>
    /// Reads back result matrix C from GPU device-resident VRAM into host memory.
    /// </summary>
    public static void ReadbackBufferC(Span<float> c, int m, int n)
    {
        s_default.ReadbackBufferC(c, m, n);
    }

    /// <summary>
    /// Executes GEMM purely in GPU device-resident VRAM without PCIe host upload/readback overhead.
    /// Operates on the persistent GPU device buffers s_devA, s_devB, s_devC.
    /// </summary>
    public static bool ExecuteResident(int m, int k, int n) => s_default.ExecuteResident(m, k, n, out _);

    /// <summary>
    /// Executes GEMM purely in GPU device-resident VRAM without PCIe host upload/readback overhead.
    /// Operates on the persistent GPU device buffers s_devA, s_devB, s_devC.
    /// </summary>
    public static bool ExecuteResident(int m, int k, int n, out D3D12KernelDiagnostics? diagnostics)
    {
        return s_default.ExecuteResident(m, k, n, out diagnostics);
    }

    private sealed class D3D12KernelInstance(uint? requiredVendorId, bool updateThreadDiagnostics)
    {
        private readonly uint? _requiredVendorId = requiredVendorId;
        private readonly bool _updateThreadDiagnostics = updateThreadDiagnostics;
        private readonly Lock _lock = new();
        private bool _initialized;
        private bool _available;

        private IDXGIFactory4? _factory;
        private IDXGIAdapter1? _adapter;
        private ID3D12Device? _device;
        private ID3D12CommandQueue? _queue;
        private readonly ExecutionSlot[] _slots = [new ExecutionSlot(), new ExecutionSlot()];
        private int _slotIndex;
        private ID3D12Fence? _fence;
        private ulong _fenceValue;
        private AutoResetEvent? _fenceEvent;

        private ID3D12RootSignature? _rootSig;
        private ID3D12PipelineState? _pipelineState;

        private ID3D12Resource? _devA;
        private ID3D12Resource? _devB;
        private ID3D12Resource? _devC;
        private ID3D12Resource? _uploadA;
        private ID3D12Resource? _uploadB;
        private ID3D12Resource? _readbackC;

        private ulong _capA;
        private ulong _capB;
        private ulong _capC;

        public string DeviceName { get; private set; } = string.Empty;
        public uint VendorId { get; private set; }
        public ID3D12Device? Device => _device;

        public void Cleanup()
        {
            lock (_lock)
            {
                CleanupHandlesLocked();
                _available = false;
            }
        }

        private void CleanupHandlesLocked()
        {
            try
            {
                if (_queue != null && _fence != null && _fenceValue > 0 && _fence.CompletedValue < _fenceValue)
                {
                    WaitForFenceValue(_fenceValue);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[D3D12GemmKernel] WaitForFenceValue during cleanup warning: {ex.Message}");
            }

            static void SafeDispose<T>(ref T? res) where T : class, IDisposable
            {
                if (res is not null)
                {
                    try { res.Dispose(); }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"[D3D12GemmKernel] Dispose warning for {typeof(T).Name}: {ex.Message}");
                    }
                    res = null;
                }
            }

            SafeDispose(ref _readbackC);
            SafeDispose(ref _uploadB);
            SafeDispose(ref _uploadA);
            SafeDispose(ref _devC);
            SafeDispose(ref _devB);
            SafeDispose(ref _devA);
            _capA = 0;
            _capB = 0;
            _capC = 0;

            SafeDispose(ref _pipelineState);
            SafeDispose(ref _rootSig);

            for (int i = 0; i < _slots.Length; i++)
            {
                SafeDispose(ref _slots[i].CommandList);
                SafeDispose(ref _slots[i].Allocator);
                _slots[i].FenceValue = 0;
            }

            SafeDispose(ref _fence);
            _fenceValue = 0;
            SafeDispose(ref _fenceEvent);
            SafeDispose(ref _queue);
            SafeDispose(ref _device);
            SafeDispose(ref _adapter);
            SafeDispose(ref _factory);
        }

        public bool EnsureInitialized()
        {
            if (_initialized) return _available;
            lock (_lock)
            {
                if (_initialized) return _available;
                try
                {
                    if (!OperatingSystem.IsWindows())
                    {
                        _available = false;
                        _initialized = true;
                        RecordError(D3D12ExecutionStage.Initialization, new PlatformNotSupportedException("Direct3D 12 is only supported on Windows."));
                        return false;
                    }

                    return InitializeWindowsCore();
                }
                catch (Exception ex)
                {
                    CleanupHandlesLocked();
                    _available = false;
                    _initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, ex);
                    return false;
                }
            }
        }

        [SupportedOSPlatform("windows")]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private bool InitializeWindowsCore()
        {
            IDXGIAdapter1? chosenAdapter = null;
            try
            {
                _factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
                nuint maxVram = 0;
                string? envAdapter = _requiredVendorId.HasValue
                    ? null
                    : Environment.GetEnvironmentVariable("GLACIER_GPU_ADAPTER");

                for (uint i = 0; _factory.EnumAdapters1(i, out IDXGIAdapter1 a).Success; i++)
                {
                    var desc = a.Description1;
                    if ((desc.Flags & AdapterFlags.Software) != 0)
                    {
                        a.Dispose();
                        continue;
                    }

                    if (_requiredVendorId.HasValue && desc.VendorId != _requiredVendorId.Value)
                    {
                        a.Dispose();
                        continue;
                    }

                    if (!string.IsNullOrEmpty(envAdapter) && desc.Description.Contains(envAdapter, StringComparison.OrdinalIgnoreCase))
                    {
                        chosenAdapter?.Dispose();
                        chosenAdapter = a;
                        break;
                    }

                    if (chosenAdapter == null || desc.DedicatedVideoMemory > maxVram)
                    {
                        chosenAdapter?.Dispose();
                        chosenAdapter = a;
                        maxVram = desc.DedicatedVideoMemory;
                    }
                    else
                    {
                        a.Dispose();
                    }
                }

                if (chosenAdapter == null)
                {
                    CleanupHandlesLocked();
                    _available = false;
                    _initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, new NotSupportedException(
                        _requiredVendorId.HasValue
                            ? $"No compatible Direct3D 12 hardware adapter found for vendor 0x{_requiredVendorId.Value:X4}."
                            : "No compatible Direct3D 12 hardware adapter found."));
                    return false;
                }

                _adapter = chosenAdapter;
                chosenAdapter = null;
                var chosenDesc = _adapter.Description1;
                DeviceName = chosenDesc.Description;
                VendorId = chosenDesc.VendorId;

                if (_requiredVendorId.HasValue && VendorId != _requiredVendorId.Value)
                {
                    CleanupHandlesLocked();
                    _available = false;
                    _initialized = true;
                    return false;
                }

                var hr = D3D12.D3D12CreateDevice(_adapter, FeatureLevel.Level_11_0, out _device!);
                if (!hr.Success || _device == null)
                {
                    CleanupHandlesLocked();
                    _available = false;
                    _initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, new InvalidOperationException($"D3D12CreateDevice failed with HRESULT 0x{hr.Code:X8}."));
                    return false;
                }

                var queueDesc = new CommandQueueDescription(CommandListType.Compute);
                _queue = _device.CreateCommandQueue(queueDesc);

                for (int i = 0; i < _slots.Length; i++)
                {
                    _slots[i].Allocator = _device.CreateCommandAllocator(CommandListType.Compute);
                    _slots[i].CommandList = _device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, _slots[i].Allocator!);
                    _slots[i].CommandList!.Close();
                    _slots[i].FenceValue = 0;
                }
                _slotIndex = 0;

                _fence = _device.CreateFence(0);
                _fenceValue = 0;
                _fenceEvent = new AutoResetEvent(false);

                // Build root signature
                var rootParams = new RootParameter[]
                {
                    new(new RootConstants(0, 0, 4), ShaderVisibility.All),
                    new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
                };

                _rootSig = _device.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rootParams), RootSignatureVersion.Version1);

                // Load cached .cso bytecode or compile HLSL once
                byte[] bytecode = LoadOrCompileShaderBytecode();
                if (bytecode.Length == 0)
                {
                    CleanupHandlesLocked();
                    _available = false;
                    _initialized = true;
                    RecordError(D3D12ExecutionStage.Initialization, new InvalidOperationException("Failed to compile GEMM HLSL compute shader bytecode."));
                    return false;
                }

                var psoDesc = new ComputePipelineStateDescription
                {
                    RootSignature = _rootSig,
                    ComputeShader = bytecode
                };
                _pipelineState = _device.CreateComputePipelineState(psoDesc);

                _available = true;
            }
            catch (Exception ex)
            {
                if (chosenAdapter is not null)
                {
                    try { chosenAdapter.Dispose(); }
                    catch (Exception disposeEx)
                    {
                        System.Diagnostics.Trace.TraceWarning($"[D3D12GemmKernel] Adapter dispose warning: {disposeEx.Message}");
                    }
                }
                CleanupHandlesLocked();
                _available = false;
                RecordError(D3D12ExecutionStage.Initialization, ex);
            }
            finally
            {
                _initialized = true;
            }

            return _available;
        }

        public D3D12KernelDiagnostics RecordError(
            D3D12ExecutionStage stage,
            Exception ex,
            int m = 0, int k = 0, int n = 0,
            ulong bytesA = 0, ulong bytesB = 0, ulong bytesC = 0,
            uint gridX = 0, uint gridY = 0)
        {
            bool isDeviceRemoved = false;
            int? removalHResult = null;
            string? removalDesc = null;

            if (_device != null)
            {
                try
                {
                    var reason = _device.DeviceRemovedReason;
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

            if (_updateThreadDiagnostics)
            {
                LastError = diag;
            }

            GlacierDiagnostics.LogError(
                $"[D3D12GemmKernel] Error at stage '{stage}' on device '{DeviceName}' " +
                $"(GEMM [{m}x{k}] * [{k}x{n}] -> [{m}x{n}], Dispatch [{gridX},{gridY},1], " +
                $"DeviceRemoved: {isDeviceRemoved}" +
                (isDeviceRemoved ? $" [0x{removalHResult:X8}: {removalDesc}]" : "") +
                $"): {ex.GetType().Name} - {ex.Message}", ex);

            return diag;
        }

        private ExecutionSlot AcquireExecutionSlot(out ID3D12GraphicsCommandList cmdList)
        {
            var slot = _slots[_slotIndex];
            _slotIndex ^= 1;
            if (slot.FenceValue > 0 && _fence!.CompletedValue < slot.FenceValue)
            {
                WaitForFenceValue(slot.FenceValue);
            }

            slot.Allocator!.Reset();
            slot.CommandList!.Reset(slot.Allocator, null);
            cmdList = slot.CommandList;
            return slot;
        }

        private void EnsureBuffers(ulong bytesA, ulong bytesB, ulong bytesC)
        {
            const ulong initCap = 16 * 1024 * 1024; // 16 MB persistent GPU heap
            if (_devA == null || _capA < bytesA)
            {
                _devA?.Dispose();
                _uploadA?.Dispose();
                ulong grown = _capA > (ulong.MaxValue / 3) * 2 ? bytesA : checked(_capA + (_capA >> 1));
                _capA = Math.Max(bytesA, Math.Max(initCap, grown));
                _devA = CreateDeviceBuffer(_capA);
                _uploadA = CreateUploadBuffer(_capA);
            }

            if (_devB == null || _capB < bytesB)
            {
                _devB?.Dispose();
                _uploadB?.Dispose();
                ulong grown = _capB > (ulong.MaxValue / 3) * 2 ? bytesB : checked(_capB + (_capB >> 1));
                _capB = Math.Max(bytesB, Math.Max(initCap, grown));
                _devB = CreateDeviceBuffer(_capB);
                _uploadB = CreateUploadBuffer(_capB);
            }

            if (_devC == null || _capC < bytesC)
            {
                _devC?.Dispose();
                _readbackC?.Dispose();
                ulong grown = _capC > (ulong.MaxValue / 3) * 2 ? bytesC : checked(_capC + (_capC >> 1));
                _capC = Math.Max(bytesC, Math.Max(initCap, grown));
                _devC = CreateDeviceBuffer(_capC, ResourceFlags.AllowUnorderedAccess);
                _readbackC = CreateReadbackBuffer(_capC);
            }
        }

        private ID3D12Resource CreateDeviceBuffer(ulong sizeInBytes, ResourceFlags flags = ResourceFlags.None)
        {
            var desc = ResourceDescription.Buffer(sizeInBytes, flags);
            var heapProps = new HeapProperties(HeapType.Default);
            return _device!.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.Common);
        }

        private ID3D12Resource CreateUploadBuffer(ulong sizeInBytes)
        {
            var desc = ResourceDescription.Buffer(sizeInBytes);
            var heapProps = new HeapProperties(HeapType.Upload);
            return _device!.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.GenericRead);
        }

        private ID3D12Resource CreateReadbackBuffer(ulong sizeInBytes)
        {
            var desc = ResourceDescription.Buffer(sizeInBytes);
            var heapProps = new HeapProperties(HeapType.Readback);
            return _device!.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.CopyDest);
        }

        private void WaitForFenceValue(ulong targetFenceValue)
        {
            if (_fence!.CompletedValue < targetFenceValue)
            {
                _fence.SetEventOnCompletion(targetFenceValue, _fenceEvent!);
                if (!_fenceEvent!.WaitOne(FenceTimeoutMilliseconds))
                {
                    throw new TimeoutException($"Direct3D 12 GPU fence synchronization timed out after {FenceTimeoutMilliseconds} ms.");
                }
            }
        }

        private void Synchronize(ExecutionSlot slot)
        {
            _fenceValue++;
            _queue!.Signal(_fence!, _fenceValue);
            slot.FenceValue = _fenceValue;
            WaitForFenceValue(_fenceValue);
        }

        public bool Execute(Tensor<float> a, Tensor<float> b, Tensor<float> c, out D3D12KernelDiagnostics? diagnostics)
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

                bytesA = checked((ulong)m * (ulong)k * sizeof(float));
                bytesB = checked((ulong)k * (ulong)n * sizeof(float));
                bytesC = checked((ulong)m * (ulong)n * sizeof(float));

                gridX = (uint)(((long)n + 63L) / 64L);
                gridY = (uint)(((long)m + 63L) / 64L);

                stage = D3D12ExecutionStage.Initialization;
                if (!EnsureInitialized() || (_requiredVendorId.HasValue && VendorId != _requiredVendorId.Value))
                {
                    diagnostics = RecordError(stage, new InvalidOperationException("Direct3D 12 initialization failed or device not available on this platform."), m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
                    return false;
                }

                if (m == 0 || k == 0 || n == 0)
                {
                    if (m > 0 && n > 0)
                    {
                        c.Fill(0.0f);
                    }

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

                lock (_lock)
                {
                    stage = D3D12ExecutionStage.BufferAllocation;
                    EnsureBuffers(bytesA, bytesB, bytesC);

                    stage = D3D12ExecutionStage.HostUpload;
                    fixed (float* pA = a.AsSpan(), pB = b.AsSpan())
                    {
                        void* pUpA = null;
                        _uploadA!.Map(0, null, &pUpA);
                        Buffer.MemoryCopy(pA, pUpA, bytesA, bytesA);
                        _uploadA.Unmap(0);

                        void* pUpB = null;
                        _uploadB!.Map(0, null, &pUpB);
                        Buffer.MemoryCopy(pB, pUpB, bytesB, bytesB);
                        _uploadB.Unmap(0);
                    }

                    stage = D3D12ExecutionStage.CommandRecording;
                    var slot = AcquireExecutionSlot(out var cmdList);

                    // Copy to device buffers
                    cmdList.ResourceBarrierTransition(_devA!, ResourceStates.Common, ResourceStates.CopyDest);
                    cmdList.ResourceBarrierTransition(_devB!, ResourceStates.Common, ResourceStates.CopyDest);
                    cmdList.CopyBufferRegion(_devA!, 0, _uploadA!, 0, bytesA);
                    cmdList.CopyBufferRegion(_devB!, 0, _uploadB!, 0, bytesB);

                    cmdList.ResourceBarrierTransition(_devA!, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
                    cmdList.ResourceBarrierTransition(_devB!, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
                    cmdList.ResourceBarrierTransition(_devC!, ResourceStates.Common, ResourceStates.UnorderedAccess);

                    // Bind pipeline and root signature
                    cmdList.SetComputeRootSignature(_rootSig!);
                    cmdList.SetPipelineState(_pipelineState!);

                    uint* pConsts = stackalloc uint[4];
                    pConsts[0] = (uint)m;
                    pConsts[1] = (uint)k;
                    pConsts[2] = (uint)n;
                    pConsts[3] = 0;
                    cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

                    cmdList.SetComputeRootShaderResourceView(1, _devA!.GPUVirtualAddress);
                    cmdList.SetComputeRootShaderResourceView(2, _devB!.GPUVirtualAddress);
                    cmdList.SetComputeRootUnorderedAccessView(3, _devC!.GPUVirtualAddress);

                    stage = D3D12ExecutionStage.Dispatch;
                    cmdList.Dispatch(gridX, gridY, 1);

                    // Transition C for readback and revert A, B
                    cmdList.ResourceBarrierTransition(_devC!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmdList.ResourceBarrierTransition(_devA!, ResourceStates.NonPixelShaderResource, ResourceStates.Common);
                    cmdList.ResourceBarrierTransition(_devB!, ResourceStates.NonPixelShaderResource, ResourceStates.Common);

                    cmdList.CopyBufferRegion(_readbackC!, 0, _devC!, 0, bytesC);
                    cmdList.ResourceBarrierTransition(_devC!, ResourceStates.CopySource, ResourceStates.Common);

                    stage = D3D12ExecutionStage.QueueExecution;
                    cmdList.Close();
                    _queue!.ExecuteCommandList(cmdList);

                    stage = D3D12ExecutionStage.FenceSynchronization;
                    Synchronize(slot);

                    stage = D3D12ExecutionStage.DeviceReadback;
                    fixed (float* pC = c.AsSpan())
                    {
                        void* pRead = null;
                        _readbackC!.Map(0, null, &pRead);
                        Buffer.MemoryCopy(pRead, pC, bytesC, bytesC);
                        _readbackC.Unmap(0);
                    }

                    if (_updateThreadDiagnostics)
                    {
                        t_lastError = null;
                    }
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

        public void UploadBuffers(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int m, int k, int n)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(m);
            ArgumentOutOfRangeException.ThrowIfNegative(k);
            ArgumentOutOfRangeException.ThrowIfNegative(n);

            ulong bytesA = checked((ulong)m * (ulong)k * sizeof(float));
            ulong bytesB = checked((ulong)k * (ulong)n * sizeof(float));
            ulong bytesC = checked((ulong)m * (ulong)n * sizeof(float));

            if (!EnsureInitialized())
                throw new InvalidOperationException("Direct3D 12 device is not available.");

            lock (_lock)
            {
                EnsureBuffers(bytesA, bytesB, bytesC);
                if (bytesA == 0 && bytesB == 0)
                {
                    return;
                }

                fixed (float* pA = a, pB = b)
                {
                    if (bytesA > 0)
                    {
                        void* pUpA = null;
                        _uploadA!.Map(0, null, &pUpA);
                        Buffer.MemoryCopy(pA, pUpA, bytesA, bytesA);
                        _uploadA.Unmap(0);
                    }

                    if (bytesB > 0)
                    {
                        void* pUpB = null;
                        _uploadB!.Map(0, null, &pUpB);
                        Buffer.MemoryCopy(pB, pUpB, bytesB, bytesB);
                        _uploadB.Unmap(0);
                    }
                }

                var slot = AcquireExecutionSlot(out var cmdList);

                TransitionUpload(cmdList, bytesA, bytesB);

                cmdList.Close();
                _queue!.ExecuteCommandList(cmdList);
                Synchronize(slot);
            }
        }

        private void TransitionUpload(ID3D12GraphicsCommandList cmdList, ulong bytesA, ulong bytesB)
        {
            cmdList.ResourceBarrierTransition(_devA!, ResourceStates.Common, ResourceStates.CopyDest);
            cmdList.ResourceBarrierTransition(_devB!, ResourceStates.Common, ResourceStates.CopyDest);
            if (bytesA > 0) cmdList.CopyBufferRegion(_devA!, 0, _uploadA!, 0, bytesA);
            if (bytesB > 0) cmdList.CopyBufferRegion(_devB!, 0, _uploadB!, 0, bytesB);
            cmdList.ResourceBarrierTransition(_devA!, ResourceStates.CopyDest, ResourceStates.Common);
            cmdList.ResourceBarrierTransition(_devB!, ResourceStates.CopyDest, ResourceStates.Common);
        }

        public void ReadbackBufferC(Span<float> c, int m, int n)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(m);
            ArgumentOutOfRangeException.ThrowIfNegative(n);

            ulong bytesC = checked((ulong)m * (ulong)n * sizeof(float));
            if (_devC == null || _readbackC == null)
                throw new InvalidOperationException("GPU device buffers not allocated.");
            if (bytesC == 0)
                return;

            lock (_lock)
            {
                var slot = AcquireExecutionSlot(out var cmdList);

                cmdList.ResourceBarrierTransition(_devC!, ResourceStates.Common, ResourceStates.CopySource);
                cmdList.CopyBufferRegion(_readbackC!, 0, _devC!, 0, bytesC);
                cmdList.ResourceBarrierTransition(_devC!, ResourceStates.CopySource, ResourceStates.Common);

                cmdList.Close();
                _queue!.ExecuteCommandList(cmdList);
                Synchronize(slot);

                fixed (float* pC = c)
                {
                    void* pRead = null;
                    _readbackC!.Map(0, null, &pRead);
                    Buffer.MemoryCopy(pRead, pC, bytesC, bytesC);
                    _readbackC.Unmap(0);
                }
            }
        }

        public bool ExecuteResident(int m, int k, int n, out D3D12KernelDiagnostics? diagnostics)
        {
            diagnostics = null;
            D3D12ExecutionStage stage = D3D12ExecutionStage.ArgumentValidation;
            if (m < 0 || k < 0 || n < 0)
            {
                diagnostics = RecordError(stage, new ArgumentOutOfRangeException(nameof(m), "Matrix dimensions must be non-negative."), m, k, n);
                return false;
            }

            ulong bytesA;
            ulong bytesB;
            ulong bytesC;
            uint gridX;
            uint gridY;
            try
            {
                bytesA = checked((ulong)m * (ulong)k * sizeof(float));
                bytesB = checked((ulong)k * (ulong)n * sizeof(float));
                bytesC = checked((ulong)m * (ulong)n * sizeof(float));
                gridX = (uint)(((long)n + 63L) / 64L);
                gridY = (uint)(((long)m + 63L) / 64L);
            }
            catch (OverflowException ex)
            {
                diagnostics = RecordError(stage, ex, m, k, n);
                return false;
            }

            try
            {
                stage = D3D12ExecutionStage.Initialization;
                if (!EnsureInitialized())
                {
                    diagnostics = RecordError(stage, new InvalidOperationException("Direct3D 12 initialization failed or device not available on this platform."), m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
                    return false;
                }

                if (m == 0 || k == 0 || n == 0)
                {
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

                lock (_lock)
                {
                    stage = D3D12ExecutionStage.BufferAllocation;
                    EnsureBuffers(bytesA, bytesB, bytesC);

                    stage = D3D12ExecutionStage.CommandRecording;
                    var slot = AcquireExecutionSlot(out var cmdList);

                    cmdList.ResourceBarrierTransition(_devA!, ResourceStates.Common, ResourceStates.NonPixelShaderResource);
                    cmdList.ResourceBarrierTransition(_devB!, ResourceStates.Common, ResourceStates.NonPixelShaderResource);
                    cmdList.ResourceBarrierTransition(_devC!, ResourceStates.Common, ResourceStates.UnorderedAccess);

                    cmdList.SetComputeRootSignature(_rootSig!);
                    cmdList.SetPipelineState(_pipelineState!);

                    uint* pConsts = stackalloc uint[4];
                    pConsts[0] = (uint)m;
                    pConsts[1] = (uint)k;
                    pConsts[2] = (uint)n;
                    pConsts[3] = 0;
                    cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

                    cmdList.SetComputeRootShaderResourceView(1, _devA!.GPUVirtualAddress);
                    cmdList.SetComputeRootShaderResourceView(2, _devB!.GPUVirtualAddress);
                    cmdList.SetComputeRootUnorderedAccessView(3, _devC!.GPUVirtualAddress);

                    stage = D3D12ExecutionStage.Dispatch;
                    cmdList.Dispatch(gridX, gridY, 1);

                    cmdList.ResourceBarrierTransition(_devC!, ResourceStates.UnorderedAccess, ResourceStates.Common);
                    cmdList.ResourceBarrierTransition(_devA!, ResourceStates.NonPixelShaderResource, ResourceStates.Common);
                    cmdList.ResourceBarrierTransition(_devB!, ResourceStates.NonPixelShaderResource, ResourceStates.Common);

                    stage = D3D12ExecutionStage.QueueExecution;
                    cmdList.Close();
                    _queue!.ExecuteCommandList(cmdList);

                    stage = D3D12ExecutionStage.FenceSynchronization;
                    Synchronize(slot);

                    if (_updateThreadDiagnostics)
                    {
                        t_lastError = null;
                    }
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

        public bool ExecuteResident(
            ulong gpuVaA,
            ulong gpuVaB,
            ulong gpuVaC,
            int m,
            int k,
            int n,
            out D3D12KernelDiagnostics? diagnostics)
        {
            diagnostics = null;
            D3D12ExecutionStage stage = D3D12ExecutionStage.ArgumentValidation;
            ulong bytesA;
            ulong bytesB;
            ulong bytesC;
            uint gridX;
            uint gridY;
            try
            {
                if (m < 0 || k < 0 || n < 0)
                {
                    diagnostics = RecordError(stage, new ArgumentOutOfRangeException(nameof(m), "Matrix dimensions must be non-negative."), m, k, n);
                    return false;
                }

                bytesA = checked((ulong)m * (ulong)k * sizeof(float));
                bytesB = checked((ulong)k * (ulong)n * sizeof(float));
                bytesC = checked((ulong)m * (ulong)n * sizeof(float));
                gridX = (uint)(((long)n + 63L) / 64L);
                gridY = (uint)(((long)m + 63L) / 64L);
            }
            catch (OverflowException ex)
            {
                diagnostics = RecordError(stage, ex, m, k, n);
                return false;
            }

            if (gpuVaA == 0 || gpuVaB == 0 || gpuVaC == 0)
            {
                diagnostics = RecordError(stage, new ArgumentException("GPU Virtual Addresses must be non-zero."), m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
                return false;
            }

            try
            {
                stage = D3D12ExecutionStage.Initialization;
                if (!EnsureInitialized())
                {
                    diagnostics = RecordError(stage, new InvalidOperationException("Direct3D 12 initialization failed or device not available on this platform."), m, k, n, bytesA, bytesB, bytesC, gridX, gridY);
                    return false;
                }

                if (m == 0 || k == 0 || n == 0)
                {
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

                lock (_lock)
                {
                    stage = D3D12ExecutionStage.CommandRecording;
                    var slot = AcquireExecutionSlot(out var cmdList);

                    // Bind compute root signature and pipeline
                    cmdList.SetComputeRootSignature(_rootSig!);
                    cmdList.SetPipelineState(_pipelineState!);

                    uint* pConsts = stackalloc uint[4];
                    pConsts[0] = (uint)m;
                    pConsts[1] = (uint)k;
                    pConsts[2] = (uint)n;
                    pConsts[3] = 0;
                    cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

                    // Direct root descriptors to VRAM buffers
                    cmdList.SetComputeRootShaderResourceView(1, gpuVaA);
                    cmdList.SetComputeRootShaderResourceView(2, gpuVaB);
                    cmdList.SetComputeRootUnorderedAccessView(3, gpuVaC);

                    stage = D3D12ExecutionStage.Dispatch;
                    cmdList.Dispatch(gridX, gridY, 1);

                    stage = D3D12ExecutionStage.QueueExecution;
                    cmdList.Close();
                    _queue!.ExecuteCommandList(cmdList);

                    stage = D3D12ExecutionStage.FenceSynchronization;
                    Synchronize(slot);

                    if (_updateThreadDiagnostics)
                    {
                        t_lastError = null;
                    }
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
}
