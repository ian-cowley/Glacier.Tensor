using System;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Autograd;

public enum AutogradOp : int
{
    Add = 1,
    Subtract = 2,
    Multiply = 3,
    MatMul = 4,
    ReLU = 5,
    Sigmoid = 6,
    Scale = 7,
    GELU = 8,
    RMSNorm = 9,
    CrossEntropyLoss = 10,
    Softmax = 11,
    LayerNorm = 12,
    Transpose = 13,
    Reshape = 14,
    Slice = 15,
    Contiguous = 16,
    MSELoss = 17,
    BinaryCrossEntropyLoss = 18,
    CategoricalCrossEntropyLoss = 19,
    BrierScoreLoss = 20
}

/// <summary>
/// Execution record on the backward autograd tape.
/// </summary>
public readonly struct TapeEntry
{
    public readonly AutogradOp Op;
    public readonly Tensor<float> Output;
    public readonly Tensor<float>? Input0;
    public readonly Tensor<float>? Input1;
    public readonly Tensor<float>? Input2;
    public readonly float Scalar;
    public readonly int IntParam0;
    public readonly int IntParam1;
    public readonly int IntParam2;

    public TapeEntry(
        AutogradOp op,
        Tensor<float> output,
        Tensor<float>? input0 = null,
        Tensor<float>? input1 = null,
        float scalar = 0f,
        Tensor<float>? input2 = null,
        int intParam0 = 0,
        int intParam1 = 0,
        int intParam2 = 0)
    {
        Op = op;
        Output = output;
        Input0 = input0;
        Input1 = input1;
        Input2 = input2;
        Scalar = scalar;
        IntParam0 = intParam0;
        IntParam1 = intParam1;
        IntParam2 = intParam2;
    }
}
