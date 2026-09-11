using System;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Tensor.Core;
using Glacier.Tensor.Interop;
using Xunit;

namespace Glacier.Tensor.Tests;

public class PolarisInteropTests
{
    [Fact]
    public void DataFrame_ToTensor_ExtractsColumnsAccurately()
    {
        var col1 = new Float32Series("f1", 3);
        col1.Memory.Span[0] = 1.0f;
        col1.Memory.Span[1] = 2.0f;
        col1.Memory.Span[2] = 3.0f;

        var col2 = new Float32Series("f2", 3);
        col2.Memory.Span[0] = 10.0f;
        col2.Memory.Span[1] = 20.0f;
        col2.Memory.Span[2] = 30.0f;

        var df = new DataFrame(new ISeries[] { col1, col2 });

        using var tensor = df.ToTensor("f1", "f2");

        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(2, tensor.Shape[1]);

        Assert.Equal(1.0f, tensor[0, 0]);
        Assert.Equal(10.0f, tensor[0, 1]);
        Assert.Equal(2.0f, tensor[1, 0]);
        Assert.Equal(20.0f, tensor[1, 1]);
        Assert.Equal(3.0f, tensor[2, 0]);
        Assert.Equal(30.0f, tensor[2, 1]);
    }

    [Fact]
    public void Tensor_ToSeries_ProducesPolarisSeries()
    {
        using var t = Tensor<float>.FromArray([5.5f, 6.5f, 7.5f], 3);
        var series = t.ToSeries("my_series");

        Assert.Equal("my_series", series.Name);
        Assert.Equal(3, series.Length);
        Assert.Equal(5.5f, series.Memory.Span[0]);
        Assert.Equal(6.5f, series.Memory.Span[1]);
        Assert.Equal(7.5f, series.Memory.Span[2]);
    }
}
