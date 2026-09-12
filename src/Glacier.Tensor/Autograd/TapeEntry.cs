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
    CrossEntropyLoss = 10
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
    public readonly float Scalar;

    public TapeEntry(AutogradOp op, Tensor<float> output, Tensor<float>? input0 = null, Tensor<float>? input1 = null, float scalar = 0f)
    {
        Op = op;
        Output = output;
        Input0 = input0;
        Input1 = input1;
        Scalar = scalar;
    }
}
