// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>One vertex of a scene mesh: clip-space position and a PARAM0 vector.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly record struct SceneVertex(float X, float Y, float Z, float W, float R, float G, float B, float A)
{
    public static SceneVertex At(float x, float y, float z, float r, float g, float b, float a = 1f) => new(x, y, z, 1f, r, g, b, a);
}

/// <summary>The draws every scene repeats: a colour target, a mesh and its fetch programs.</summary>
internal static class SceneDraws
{
    /// <summary>
    /// A linear colour target in guest memory, cleared to <paramref name="fill"/> bytes. Rows
    /// sit at the 256-byte linear pitch; read it back with <see cref="ReadImage"/>.
    /// </summary>
    public static ulong ColorTarget(this SceneRig rig, uint width, uint height, uint bytesPerPixel = 4, byte fill = 0)
    {
        var size = (ulong)LinearPitchBytes(width, bytesPerPixel) * height;
        var address = rig.Memory.Allocate(size, 0x10000);
        if (fill != 0)
        {
            var bytes = new byte[size];
            Array.Fill(bytes, fill);
            rig.Memory.Write(address, bytes);
        }

        return address;
    }

    /// <summary>
    /// Uploads the mesh as two 16-byte-record buffers and returns the vertex program's
    /// user data: the position V# then the PARAM0 V#.
    /// </summary>
    public static uint[] UploadMesh(this SceneRig rig, IReadOnlyList<SceneVertex> vertices)
    {
        var positions = new float[vertices.Count * 4];
        var parameters = new float[vertices.Count * 4];
        for (var index = 0; index < vertices.Count; index++)
        {
            var vertex = vertices[index];
            positions[(index * 4) + 0] = vertex.X;
            positions[(index * 4) + 1] = vertex.Y;
            positions[(index * 4) + 2] = vertex.Z;
            positions[(index * 4) + 3] = vertex.W;
            parameters[(index * 4) + 0] = vertex.R;
            parameters[(index * 4) + 1] = vertex.G;
            parameters[(index * 4) + 2] = vertex.B;
            parameters[(index * 4) + 3] = vertex.A;
        }

        var positionAddress = rig.Memory.Upload<float>(positions);
        var parameterAddress = rig.Memory.Upload<float>(parameters);
        return
        [
            .. Descriptors.Buffer(positionAddress, 16, (uint)vertices.Count),
            .. Descriptors.Buffer(parameterAddress, 16, (uint)vertices.Count),
        ];
    }

    /// <summary>A full-viewport quad as two triangles at depth <paramref name="z"/>.</summary>
    public static SceneVertex[] Quad(float z, float r, float g, float b, float a = 1f,
        float left = -1f, float bottom = -1f, float right = 1f, float top = 1f) =>
    [
        SceneVertex.At(left, bottom, z, r, g, b, a),
        SceneVertex.At(right, bottom, z, r, g, b, a),
        SceneVertex.At(left, top, z, r, g, b, a),
        SceneVertex.At(left, top, z, r, g, b, a),
        SceneVertex.At(right, bottom, z, r, g, b, a),
        SceneVertex.At(right, top, z, r, g, b, a),
    ];

    /// <summary>Draws <paramref name="vertices"/> with the fetch and interpolated-colour programs.</summary>
    public static ScenePm4 DrawMesh(this SceneRig rig, ScenePm4 pm4, SceneState state, IReadOnlyList<SceneVertex> vertices,
        uint[]? pixelProgram = null, uint[]? pixelUserData = null, bool predicated = false)
    {
        var vertex = rig.Program(ScenePrograms.FetchVertex);
        var pixel = rig.Program(pixelProgram ?? ScenePrograms.InterpolatedColor);
        state.VertexShader(vertex, rig.UploadMesh(vertices));
        state.PixelShader(pixel, pixelUserData ?? []);
        state.Emit(pm4);
        pm4.NumInstances(1);
        pm4.DrawAuto((uint)vertices.Count, predicated);
        return pm4;
    }

    /// <summary>
    /// A linear image in guest memory: rows padded to the 256-byte linear pitch the
    /// hardware (and the image layout) uses. <paramref name="rows"/> holds tightly packed rows.
    /// </summary>
    public static ulong LinearImage(this SceneRig rig, uint width, uint height, uint bytesPerTexel, ReadOnlySpan<byte> rows)
    {
        var pitch = LinearPitchBytes(width, bytesPerTexel);
        var bytes = new byte[pitch * height];
        for (var y = 0; y < height; y++)
        {
            rows.Slice((int)(y * width * bytesPerTexel), (int)(width * bytesPerTexel)).CopyTo(bytes.AsSpan((int)(y * pitch)));
        }

        var address = rig.Memory.Allocate((ulong)bytes.Length, 0x10000);
        rig.Memory.Write(address, bytes);
        return address;
    }

    /// <summary>A linear image's texels as tightly packed rows (the 256-byte pitch padding removed).</summary>
    public static byte[] ReadImage(this SceneRig rig, ulong address, uint width, uint height, uint bytesPerTexel = 4)
    {
        var pitch = LinearPitchBytes(width, bytesPerTexel);
        var padded = rig.Read(address, (int)(pitch * height));
        var rowBytes = (int)(width * bytesPerTexel);
        var packed = new byte[rowBytes * height];
        for (var y = 0; y < height; y++)
        {
            padded.AsSpan((int)(y * pitch), rowBytes).CopyTo(packed.AsSpan(y * rowBytes));
        }

        return packed;
    }

    public static T[] ReadImage<T>(this SceneRig rig, ulong address, uint width, uint height) where T : unmanaged =>
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(
            rig.ReadImage(address, width, height, (uint)System.Runtime.CompilerServices.Unsafe.SizeOf<T>())).ToArray();

    public static uint LinearPitchBytes(uint width, uint bytesPerTexel) =>
        (width * bytesPerTexel + 255) / 256 * 256;

    /// <summary>A full-viewport quad whose PARAM0.xy is the texture coordinate (0,0 at the top-left).</summary>
    public static SceneVertex[] TexturedQuad(float z = 0.5f) =>
    [
        SceneVertex.At(-1, -1, z, 0, 1, 0, 1),
        SceneVertex.At(1, -1, z, 1, 1, 0, 1),
        SceneVertex.At(-1, 1, z, 0, 0, 0, 1),
        SceneVertex.At(-1, 1, z, 0, 0, 0, 1),
        SceneVertex.At(1, -1, z, 1, 1, 0, 1),
        SceneVertex.At(1, 1, z, 1, 0, 0, 1),
    ];

    /// <summary>Dispatches <paramref name="program"/> with the given user data.</summary>
    public static ScenePm4 Dispatch(this SceneRig rig, ScenePm4 pm4, uint[] program, uint[] userData, uint groupsX,
        uint groupsY = 1, uint groupsZ = 1, uint threadsX = 64, uint threadsY = 1, uint threadsZ = 1, uint ldsBytes = 0,
        bool groupIdY = false, bool groupIdZ = false, uint threadIdComponents = 0)
    {
        var state = new SceneState(1, 1);
        state.Context.Clear();
        state.Uconfig.Clear();
        state.ComputeShader(rig.Program(program), threadsX, threadsY, threadsZ, userData, true, groupIdY, groupIdZ,
            threadIdComponents, ldsBytes);
        state.Emit(pm4);
        pm4.DispatchDirect(groupsX, groupsY, groupsZ, 1);
        return pm4;
    }

    public static uint Pixel(byte[] rgba, uint width, uint x, uint y) =>
        BitConverter.ToUInt32(rgba, (int)(((y * width) + x) * 4));

    public static string Hex(uint value) => $"0x{value:X8}";
}

/// <summary>The device gate every scene runs behind.</summary>
public abstract class SceneTestBase(ITestOutputHelper output)
{
    protected ITestOutputHelper Output { get; } = output;

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    internal SceneRig? CreateRig(bool splitWorker = false)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            Output.WriteLine("Scenes run on the device only behind the exclusive device gate (SHARPEMU_TEST_REQUIRE_DEVICE=1).");
            return null;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return new SceneRig(Device.Value, splitWorker);
    }
}
