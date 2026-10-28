// VoxLock - 模型常量，Unity 移植版的各个部分共用。
//
// 这些数字是网络层的接口约定，不能改动：
//   SR / HOP / NFFT / PAD  - 16 kHz、跳长 128（8 ms）、FFT 512（32 ms）、反射填充 256
//   GAIN                   - 网络输出的说话人约放大到 -2 倍，这个系数把电平还原回输入水平
//   WindowSquared          - Hann^2，重叠相加归一化时用的分析/合成窗
using System;

namespace VoxLock
{
    /// <summary>
    /// 加载 <c>Assets/StreamingAssets</c> 下哪一对 ONNX 文件。文件名沿用 Python 包在 <c>onnx/v2</c> 里
    /// 的命名；音质和 CPU 开销取自 VoxLock 的 README。
    /// </summary>
    public enum VoxLockPrecision
    {
        /// <summary>enroll.onnx + chunk_int8.onnx - 74 MB，音质为 fp32 的 99.8%，速度最快。默认。</summary>
        Int8 = 0,
        /// <summary>enroll.onnx + chunk_int8_full.onnx - 71 MB，LSTM 和矩阵乘也是 8 位，音质 99.1%。</summary>
        Int8Full = 1,
        /// <summary>enroll_fp16.onnx + chunk_fp16.onnx - 47 MB，float32 权重舍入到 float16。</summary>
        Fp16 = 2,
        /// <summary>enroll.onnx + chunk.onnx - 94 MB，用来对标评测的模型。</summary>
        Fp32 = 3,
    }

    public static class VoxLockConfig
    {
        public const int SampleRate = 16000;
        public const int Hop = 128;
        public const int FftSize = 512;
        public const int Pad = 256;

        /// <summary>
        /// StreamingAssets 里存放 ONNX 文件的文件夹，相对于 StreamingAssets，结尾不带斜杠。
        /// 空字符串表示 StreamingAssets 本身。v2 的图放在 "voxlock/v2"。
        /// </summary>
        public const string ModelFolder = "voxlock/v2";

        /// <summary>网络输出的说话人约放大到 -2 倍，这个系数把电平还原回输入水平。</summary>
        public const float Gain = -0.5f;

        /// <summary>网络要求的声音样本长度恰好是 3 s。</summary>
        public const int SampleLength = 3 * SampleRate;

        /// <summary>在评测路径里，<see cref="VoxLockAudio.LoudSample"/> 在通话开头多大的范围内找最响的片段。</summary>
        public const int LoudWithin = 15 * SampleRate;

        /// <summary>样本更长时，<see cref="VoxLockAudio.FitSample"/> 在多大的范围内找。</summary>
        public const int LoudWithinLong = 30 * SampleRate;

        /// <summary>每次模型调用的帧数。100 是参考实现用的值；调大更快，但会增加延迟。</summary>
        public const int DefaultFramesPerRun = 100;

        /// <summary>重叠相加用的分析/合成窗，已平方。</summary>
        public static readonly float[] WindowSquared = BuildWindow();

        /// <summary>10 ms，实时说话人检测统计能量所用的帧长。</summary>
        public const int EnergyFrame = 160;

        private static float[] BuildWindow()
        {
            var w = new float[FftSize];
            for (int i = 0; i < FftSize; i++)
            {
                float h = 0.5f - 0.5f * (float)Math.Cos(2.0 * Math.PI * i / FftSize);
                w[i] = h * h;
            }
            return w;
        }

        public static string EnrollFileName(VoxLockPrecision precision)
        {
            switch (precision)
            {
                case VoxLockPrecision.Fp16: return "enroll_fp16.onnx";
                default: return "enroll.onnx";
            }
        }

        public static string ChunkFileName(VoxLockPrecision precision)
        {
            switch (precision)
            {
                case VoxLockPrecision.Int8: return "chunk_int8.onnx";
                case VoxLockPrecision.Int8Full: return "chunk_int8_full.onnx";
                case VoxLockPrecision.Fp16: return "chunk_fp16.onnx";
                default: return "chunk.onnx";
            }
        }

        public static string Describe(VoxLockPrecision precision)
        {
            switch (precision)
            {
                case VoxLockPrecision.Int8: return "int8 (74 MB, 99.8% of fp32 quality)";
                case VoxLockPrecision.Int8Full: return "int8-full (71 MB, LSTMs and matmuls in 8-bit)";
                case VoxLockPrecision.Fp16: return "fp16 (47 MB, float32 weights in float16)";
                default: return "fp32 (94 MB, the benchmarked model)";
            }
        }
    }
}
