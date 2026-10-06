namespace Glacier.Tensor.Compute;

/// <summary>
/// Identifies the stage of execution during Direct3D 12 GEMM kernel operations.
/// </summary>
public enum D3D12ExecutionStage
{
    None = 0,
    Initialization,
    ArgumentValidation,
    BufferAllocation,
    HostUpload,
    CommandRecording,
    Dispatch,
    QueueExecution,
    FenceSynchronization,
    DeviceReadback
}
