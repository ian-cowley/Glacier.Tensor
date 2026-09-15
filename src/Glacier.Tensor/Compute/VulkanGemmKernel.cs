using System;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Tensor.Core;

namespace Glacier.Tensor.Compute;

/// <summary>
/// Cross-platform Vulkan 1.3+ compute kernel for general matrix multiplication (GEMM).
/// Embeds precompiled SPIR-V bytecode with a 16x16 shared-memory tiled compute kernel.
/// Zero-dependency execution across AMD Radeon, Intel Arc, and NVIDIA GPUs on Windows and Linux.
/// </summary>
public static unsafe class VulkanGemmKernel
{
    private static readonly Lock s_lock = new();
    private static bool s_initialized;
    private static bool s_available;

    private static VulkanContext? s_context;
    private static IntPtr s_shaderModule;
    private static IntPtr s_descriptorSetLayout;
    private static IntPtr s_pipelineLayout;
    private static IntPtr s_pipeline;
    private static IntPtr s_descriptorPool;
    private static IntPtr s_descriptorSet;

    private static IntPtr s_bufA, s_memA;
    private static IntPtr s_bufB, s_memB;
    private static IntPtr s_bufC, s_memC;
    private static ulong s_capA;
    private static ulong s_capB;
    private static ulong s_capC;

    public static string DeviceName => s_context?.DeviceName ?? string.Empty;
    public static bool IsSupported => EnsureInitialized();

    /// <summary>
    /// Base64-encoded precompiled SPIR-V bytecode (3,192 bytes, compiled with glslc -O).
    /// </summary>
    public const string SpvBase64 = "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";

    public static readonly byte[] SpvBytecode = Convert.FromBase64String(SpvBase64);

    public const string GlslSource = @"
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint N;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer BufA {
    float A[];
};

layout(std430, set = 0, binding = 1) readonly buffer BufB {
    float B[];
};

layout(std430, set = 0, binding = 2) writeonly buffer BufC {
    float C[];
};

shared float sA[16][16];
shared float sB[16][16];

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;

    float acc = 0.0;
    uint numTiles = (pc.K + 15) / 16;

    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * 16 + lx;
        uint bRow = t * 16 + ly;

        sA[ly][lx] = (row < pc.M && aCol < pc.K) ? A[row * pc.K + aCol] : 0.0;
        sB[ly][lx] = (bRow < pc.K && col < pc.N) ? B[bRow * pc.N + col] : 0.0;

        barrier();

        for (uint k = 0; k < 16; ++k) {
            acc += sA[ly][k] * sB[k][lx];
        }

        barrier();
    }

    if (row < pc.M && col < pc.N) {
        C[row * pc.N + col] = acc;
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
                if (!VulkanContext.IsSupported)
                {
                    s_available = false;
                    s_initialized = true;
                    return false;
                }

                s_context = new VulkanContext(0);
                var dev = s_context.DeviceHandle;

                // 1. Create Shader Module from embedded SPIR-V
                fixed (byte* pSpv = SpvBytecode)
                {
                    var smInfo = new VulkanDriver.VkShaderModuleCreateInfo
                    {
                        sType = VulkanDriver.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                        codeSize = (nuint)SpvBytecode.Length,
                        pCode = (IntPtr)pSpv
                    };
                    int smRes = VulkanDriver.CreateShaderModule(dev, ref smInfo, IntPtr.Zero, out s_shaderModule);
                    if (smRes != 0 || s_shaderModule == IntPtr.Zero)
                    {
                        s_available = false;
                        s_initialized = true;
                        return false;
                    }
                }

                // 2. Create Descriptor Set Layout (3 storage buffers: binding 0=A, 1=B, 2=C)
                var bindings = stackalloc VulkanDriver.VkDescriptorSetLayoutBinding[3];
                for (uint i = 0; i < 3; i++)
                {
                    bindings[i] = new VulkanDriver.VkDescriptorSetLayoutBinding
                    {
                        binding = i,
                        descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount = 1,
                        stageFlags = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT
                    };
                }

                var dslInfo = new VulkanDriver.VkDescriptorSetLayoutCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    bindingCount = 3,
                    pBindings = (IntPtr)bindings
                };
                int dslRes = VulkanDriver.CreateDescriptorSetLayout(dev, ref dslInfo, IntPtr.Zero, out s_descriptorSetLayout);
                if (dslRes != 0 || s_descriptorSetLayout == IntPtr.Zero)
                {
                    s_available = false;
                    s_initialized = true;
                    return false;
                }

                // 3. Create Pipeline Layout with push constants (M, K, N = 12 bytes)
                var pcRange = new VulkanDriver.VkPushConstantRange
                {
                    stageFlags = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT,
                    offset = 0,
                    size = 12
                };

                var pDsl = s_descriptorSetLayout;
                var plInfo = new VulkanDriver.VkPipelineLayoutCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    setLayoutCount = 1,
                    pSetLayouts = (IntPtr)(&pDsl),
                    pushConstantRangeCount = 1,
                    pPushConstantRanges = (IntPtr)(&pcRange)
                };
                int plRes = VulkanDriver.CreatePipelineLayout(dev, ref plInfo, IntPtr.Zero, out s_pipelineLayout);
                if (plRes != 0 || s_pipelineLayout == IntPtr.Zero)
                {
                    s_available = false;
                    s_initialized = true;
                    return false;
                }

                // 4. Create Compute Pipeline
                var pMain = Marshal.StringToHGlobalAnsi("main");
                var stageInfo = new VulkanDriver.VkPipelineShaderStageCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT,
                    module = s_shaderModule,
                    pName = pMain
                };

                var pipeInfo = new VulkanDriver.VkComputePipelineCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    stage = stageInfo,
                    layout = s_pipelineLayout
                };

                var pipes = new IntPtr[1];
                int pipeRes = VulkanDriver.CreateComputePipelines(dev, IntPtr.Zero, 1, new[] { pipeInfo }, IntPtr.Zero, pipes);
                Marshal.FreeHGlobal(pMain);

                if (pipeRes != 0 || pipes[0] == IntPtr.Zero)
                {
                    s_available = false;
                    s_initialized = true;
                    return false;
                }
                s_pipeline = pipes[0];

                // 5. Create Descriptor Pool and allocate Descriptor Set
                var poolSize = new VulkanDriver.VkDescriptorPoolSize
                {
                    type = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount = 3
                };
                var poolInfo = new VulkanDriver.VkDescriptorPoolCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets = 1,
                    poolSizeCount = 1,
                    pPoolSizes = (IntPtr)(&poolSize)
                };
                int poolRes = VulkanDriver.CreateDescriptorPool(dev, ref poolInfo, IntPtr.Zero, out s_descriptorPool);
                if (poolRes != 0 || s_descriptorPool == IntPtr.Zero)
                {
                    s_available = false;
                    s_initialized = true;
                    return false;
                }

                var setAlloc = new VulkanDriver.VkDescriptorSetAllocateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool = s_descriptorPool,
                    descriptorSetCount = 1,
                    pSetLayouts = (IntPtr)(&pDsl)
                };
                var sets = new IntPtr[1];
                int setRes = VulkanDriver.AllocateDescriptorSets(dev, ref setAlloc, sets);
                if (setRes != 0 || sets[0] == IntPtr.Zero)
                {
                    s_available = false;
                    s_initialized = true;
                    return false;
                }
                s_descriptorSet = sets[0];

                s_available = true;
            }
            catch
            {
                s_available = false;
            }
            finally
            {
                s_initialized = true;
            }

            return s_available;
        }
    }

    private static void EnsureBuffers(ulong bytesA, ulong bytesB, ulong bytesC)
    {
        uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        uint memProps = VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VulkanDriver.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        if (s_bufA == IntPtr.Zero || s_capA < bytesA)
        {
            if (s_bufA != IntPtr.Zero) s_context!.DestroyBuffer(s_bufA, s_memA);
            s_capA = Math.Max(bytesA, 1024 * 1024);
            s_context!.CreateBuffer(s_capA, usage, memProps, out s_bufA, out s_memA);
        }

        if (s_bufB == IntPtr.Zero || s_capB < bytesB)
        {
            if (s_bufB != IntPtr.Zero) s_context!.DestroyBuffer(s_bufB, s_memB);
            s_capB = Math.Max(bytesB, 1024 * 1024);
            s_context!.CreateBuffer(s_capB, usage, memProps, out s_bufB, out s_memB);
        }

        if (s_bufC == IntPtr.Zero || s_capC < bytesC)
        {
            if (s_bufC != IntPtr.Zero) s_context!.DestroyBuffer(s_bufC, s_memC);
            s_capC = Math.Max(bytesC, 1024 * 1024);
            s_context!.CreateBuffer(s_capC, usage, memProps, out s_bufC, out s_memC);
        }
    }

    public static bool Execute(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        if (!EnsureInitialized()) return false;

        int M = a.Shape[0];
        int K = a.Shape[1];
        int N = b.Shape[1];

        ulong bytesA = (ulong)(M * K * sizeof(float));
        ulong bytesB = (ulong)(K * N * sizeof(float));
        ulong bytesC = (ulong)(M * N * sizeof(float));

        lock (s_lock)
        {
            try
            {
                EnsureBuffers(bytesA, bytesB, bytesC);

                // Copy inputs to host-visible coherent buffer
                fixed (float* pA = a.AsSpan(), pB = b.AsSpan())
                {
                    s_context!.CopyToBuffer(s_memA, (IntPtr)pA, bytesA);
                    s_context!.CopyToBuffer(s_memB, (IntPtr)pB, bytesB);
                }

                // Update descriptor set with buffer references
                var bufInfos = stackalloc VulkanDriver.VkDescriptorBufferInfo[3];
                bufInfos[0] = new() { buffer = s_bufA, offset = 0, range = VulkanDriver.VK_WHOLE_SIZE };
                bufInfos[1] = new() { buffer = s_bufB, offset = 0, range = VulkanDriver.VK_WHOLE_SIZE };
                bufInfos[2] = new() { buffer = s_bufC, offset = 0, range = VulkanDriver.VK_WHOLE_SIZE };

                var writes = new VulkanDriver.VkWriteDescriptorSet[3];
                for (uint i = 0; i < 3; i++)
                {
                    writes[i] = new VulkanDriver.VkWriteDescriptorSet
                    {
                        sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet = s_descriptorSet,
                        dstBinding = i,
                        dstArrayElement = 0,
                        descriptorCount = 1,
                        descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo = (IntPtr)(&bufInfos[i])
                    };
                }
                VulkanDriver.UpdateDescriptorSets(s_context!.DeviceHandle, 3, writes, 0, IntPtr.Zero);

                // Record command buffer
                var cmd = s_context.CommandBufferHandle;
                VulkanDriver.ResetCommandBuffer(cmd, VulkanDriver.VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

                var beginInfo = new VulkanDriver.VkCommandBufferBeginInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                };
                VulkanDriver.BeginCommandBuffer(cmd, ref beginInfo);

                VulkanDriver.CmdBindPipeline(cmd, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, s_pipeline);

                VulkanDriver.CmdBindDescriptorSets(cmd, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, s_pipelineLayout, 0, 1, new[] { s_descriptorSet }, 0, IntPtr.Zero);

                uint* pPush = stackalloc uint[3];
                pPush[0] = (uint)M;
                pPush[1] = (uint)K;
                pPush[2] = (uint)N;
                VulkanDriver.CmdPushConstants(cmd, s_pipelineLayout, VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, (IntPtr)pPush);

                uint gridX = (uint)((N + 15) / 16);
                uint gridY = (uint)((M + 15) / 16);
                VulkanDriver.CmdDispatch(cmd, gridX, gridY, 1);

                VulkanDriver.EndCommandBuffer(cmd);

                // Submit and wait
                var pCmds = stackalloc IntPtr[1] { cmd };
                var submitInfo = new VulkanDriver.VkSubmitInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount = 1,
                    pCommandBuffers = (IntPtr)pCmds
                };

                VulkanDriver.QueueSubmit(s_context.QueueHandle, 1, ref submitInfo, IntPtr.Zero);
                VulkanDriver.QueueWaitIdle(s_context.QueueHandle);

                // Readback
                fixed (float* pC = c.AsSpan())
                {
                    s_context.CopyFromBuffer((IntPtr)pC, s_memC, bytesC);
                }

                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
