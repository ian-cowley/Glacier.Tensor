using Glacier.Compute;
using Glacier.Tensor.Core;
using Glacier.Tensor.Distributed;
using Xunit;

namespace Glacier.Tensor.Tests;

public class DistributedTensorTests
{
    [Fact]
    public async Task AllReduceClusterAsync_SumsTensorsAcrossCluster()
    {
        using var t0 = new Tensor<float>(2, 2);
        using var t1 = new Tensor<float>(2, 2);
        using var t2 = new Tensor<float>(2, 2);

        t0[0, 0] = 1f; t0[0, 1] = 2f; t0[1, 0] = 3f; t0[1, 1] = 4f;
        t1[0, 0] = 10f; t1[0, 1] = 20f; t1[1, 0] = 30f; t1[1, 1] = 40f;
        t2[0, 0] = 100f; t2[0, 1] = 200f; t2[1, 0] = 300f; t2[1, 1] = 400f;

        var cluster = new[] { t0, t1, t2 };

        await cluster.AllReduceClusterAsync(ReductionOp.Sum);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(111f, cluster[i][0, 0]);
            Assert.Equal(222f, cluster[i][0, 1]);
            Assert.Equal(333f, cluster[i][1, 0]);
            Assert.Equal(444f, cluster[i][1, 1]);
        }
    }
}
