namespace Glacier.Tensor.Compute;

using System;

/// <summary>
/// Structured diagnostics captured during Direct3D 12 GEMM kernel execution.
/// </summary>
public sealed class D3D12KernelDiagnostics
{
    /// <summary>
    /// Gets or sets whether execution succeeded.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the execution stage where the error occurred or was recorded.
    /// </summary>
    public D3D12ExecutionStage Stage { get; set; } = D3D12ExecutionStage.None;

    /// <summary>
    /// Gets the string representation of the execution stage.
    /// </summary>
    public string? StageName => Stage.ToString();

    /// <summary>
    /// Gets or sets the name of the Direct3D 12 device/adapter used.
    /// </summary>
    public string DeviceName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the Direct3D 12 device was removed (TDR or reset).
    /// </summary>
    public bool IsDeviceRemoved { get; set; }

    /// <summary>
    /// Gets or sets the DXGI device removal HRESULT if applicable.
    /// </summary>
    public int? DeviceRemovedReasonHResult { get; set; }

    /// <summary>
    /// Gets or sets the textual description of the device removal reason.
    /// </summary>
    public string? DeviceRemovedReasonDescription { get; set; }

    /// <summary>
    /// Matrix row dimension M.
    /// </summary>
    public int M { get; set; }

    /// <summary>
    /// Matrix inner dimension K.
    /// </summary>
    public int K { get; set; }

    /// <summary>
    /// Matrix column dimension N.
    /// </summary>
    public int N { get; set; }

    /// <summary>
    /// Byte capacity required for operand tensor A.
    /// </summary>
    public ulong BytesA { get; set; }

    /// <summary>
    /// Byte capacity required for operand tensor B.
    /// </summary>
    public ulong BytesB { get; set; }

    /// <summary>
    /// Byte capacity required for result tensor C.
    /// </summary>
    public ulong BytesC { get; set; }

    /// <summary>
    /// Compute shader threadgroup dispatch dimension X.
    /// </summary>
    public uint GridX { get; set; }

    /// <summary>
    /// Compute shader threadgroup dispatch dimension Y.
    /// </summary>
    public uint GridY { get; set; }

    /// <summary>
    /// Gets or sets the error message.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the exception message (alias for ErrorMessage).
    /// </summary>
    public string? ExceptionMessage
    {
        get => ErrorMessage;
        set => ErrorMessage = value;
    }

    /// <summary>
    /// Gets or sets the type name of the caught exception.
    /// </summary>
    public string? ExceptionType { get; set; }

    /// <summary>
    /// Gets or sets the caught exception if available.
    /// </summary>
    public Exception? Exception { get; set; }

    /// <summary>
    /// Gets or sets the HRESULT associated with the exception or failure.
    /// </summary>
    public int? HResult { get; set; }

    /// <summary>
    /// Gets or sets the stack trace of the failure.
    /// </summary>
    public string? StackTrace { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when the diagnostic was captured.
    /// </summary>
    public DateTimeOffset Timestamp { get; set; } = DateTimeOffset.UtcNow;
}
