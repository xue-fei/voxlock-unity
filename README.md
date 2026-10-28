# voxlock-unity

[VoxLock](https://github.com/arbazsiddiqui/voxlock) 的 Unity C# 移植版，基于
[onnxruntime-unity](https://github.com/xue-fei/onnxruntime-unity) 与 CPU 插件
[onnxruntime-unity-cpu](https://github.com/xue-fei/onnxruntime-unity-cpu)。

VoxLock 是一个语音隔离器：输入通话音频，它自己找出主叫，把房间里的其他人声去掉。这个移植版逐样本复现了
Python 包的管线——fp32 模型上两者一致到 float32 精度。

这里没有播放，也不采集麦克风，只有隔离管线本身，以及把文件喂给它的两种方式（整文件 / 流式）。

## 目录结构

| 路径 | |
|---|---|
| `Assets/Scripts/VoxLock/` | 移植核心：模型加载、ONNX 会话、流式处理、WAV 读写、对外 API |
| `Assets/Scripts/VoxLockDemo/` | 演示组件：选一个 WAV、选一个模式、跑一遍、看进度 |
| `Assets/StreamingAssets/voxlock/v2/` | ONNX 模型文件 |
| `Assets/Audio/` | 示例音频 |
| `Assets/Isolated/` | Demo 默认输出目录 |

## 用法

```csharp
using VoxLock;

// 加载一次。
var session = VoxLockSession.Create(
    VoxLockModelLoader.LoadBytes("enroll.onnx"),
    VoxLockModelLoader.LoadBytes("chunk_int8.onnx"),
    VoxLockPrecision.Int8, threads: 2);

// 整通电话进，主叫出，长度不变。声音样本取前 15 秒里最响的 3 秒。
float[] caller = VoxLockAudio.Isolate(session, call);

// 流式：从第一块就开始隔离，比输入晚 24 ms。
using (var stream = VoxLockAudio.CreateStream(session, sample))
{
    var output = new float[VoxLockStream.OutputCapacityFor(chunk.Length)];
    int n = stream.Push(chunk, chunk.Length, output);
    int tail = stream.Flush(flushBuffer);   // 用 stream.FlushCapacity 定大小
}

// 或者让它自己找主叫：听够 3 秒语音之前原样透传，之后开始隔离。
using (var live = VoxLockAudio.CreateLive(session, seconds: 3f))
{
    int n = live.Push(chunk, chunk.Length, output);
}
```

所有接口都接收 16 kHz 单声道 float32。处理文件用 `VoxLockAudio.IsolateFile`（整文件）或
`IsolateFileStreaming`（20 ms 一块，内存占用与文件长度无关），两者都直接读写 WAV，非 16 kHz 会重采样、
立体声会下混。

注意：一次 push 最多可能比输入多返回 512 个样本，缓冲要用 `OutputCapacityFor(count)` 定大小，别直接用
`count`。

## Demo

把 `VoxLockDemo` 挂到任意 GameObject 上。它从 StreamingAssets 加载模型，然后按 `RunMode` 处理
`InputWav`（绝对路径，或 `Assets` 下的相对路径），结果写到 `Assets/Isolated/<原名>_isolated.wav`——
放在 `Assets` 下，Unity 会导成 AudioClip，点一下就能试听。

**输入要选有两个人的音频。** VoxLock 去掉的是其他人的声音，单说话人的文件没有可移除的对象，输出会几乎
和输入一样。`Assets/Audio/bystander_off.wav` 是两人同时说话（14 秒，官方 demo 素材），
`bystander_reference.wav` 是它该有的输出；`talker.wav` 是单人说话，拿来对比正好能看出这个区别。

Demo 会报告 `Similarity`（输入输出的相关系数）：0.76 左右说明第二个人声被分离掉了，0.98 以上说明没有东西
可移除。这比用耳朵判断可靠。

## 精度

| 精度 | 文件 | 体积 | 说明 |
|---|---|---|---|
| `Int8`（默认） | `enroll.onnx` + `chunk_int8.onnx` | 74 MB | 最快 |
| `Int8Full` | `enroll.onnx` + `chunk_int8_full.onnx` | 71 MB | LSTM 和矩阵乘都走 8 位 |
| `Fp16` | `enroll_fp16.onnx` + `chunk_fp16.onnx` | 47 MB | 体积最小 |
| `Fp32` | `enroll.onnx` + `chunk.onnx` | 94 MB | 参与评测的模型 |

int8 图不是 bit-exact 的：动态量化会从每次调用里各帧的取值范围推导激活 scale，所以分块大小会轻微改变结果
（~5e-3）。需要完全可复现就用 `Fp32`。

## 许可

VoxLock 是 Apache-2.0。本移植版沿用同一许可，见 `LICENSE`。
