// VoxLock - 对外入口，与 Python 包的 API 一一对应。
//
//   VoxLockAudio.LoudSample(mix, 3, 15)      - 前 15 s 里最响的 3 s：VoxLock 就是这样评测的
//   VoxLockAudio.FitSample(sample)           - 任意不短于 3 s 的样本，裁到正好 3 s
//   VoxLockAudio.Isolate(session, audio)     - 整通电话进，说话人出，长度不变
//   VoxLockAudio.CreateStream(session, smp)  - 从第一块就开始隔离，用于已有样本的场景
//   VoxLockAudio.CreateLive(session, 3)      - 听够 3 s 语音之前原样透传
//
// 文件层面有两条便捷路径，都直接读写 WAV，不需要自己处理格式：
//   VoxLockAudio.IsolateFile(session, in, out)    - 整文件处理，一次读入、一次写出
//   VoxLockAudio.IsolateFileStreaming(session, in, out) - 按块流式处理，内存占用与文件长度无关
//
// 全部接收 16 kHz 单声道 float32。命名空间是 VoxLock，类型也叫 VoxLockAudio，所以静态入口写作
// VoxLock.VoxLockAudio.X——和 Python 包里的 voxlock.VoxLock 方法拼法一致，同时避免命名空间内部的名字冲突。
using System;
using System.IO;

namespace VoxLock
{
    public static class VoxLockAudio
    {
        /// <summary>
        /// VoxLock 评测时用的声音样本：通话前 <paramref name="within"/> 秒内最响的
        /// <paramref name="seconds"/> 秒。
        /// </summary>
        public static float[] LoudSample(float[] mix, float seconds = 3f, float within = 15f)
        {
            if (mix == null) throw new ArgumentNullException(nameof(mix));

            int frame = VoxLockConfig.EnergyFrame;
            int limit = Math.Min(mix.Length, (int)(within * VoxLockConfig.SampleRate));
            int n = limit / frame;
            int w = (int)(seconds * VoxLockConfig.SampleRate / frame);
            if (n <= w) return Sub(mix, 0, limit);

            // 先算每个 10 ms 帧的均方，再取滑动窗口和的最大值位置。
            var energy = new float[n];
            for (int i = 0; i < n; i++)
            {
                float sum = 0f;
                int at = i * frame;
                for (int j = 0; j < frame; j++)
                {
                    float v = mix[at + j];
                    sum += v * v;
                }
                energy[i] = sum / frame;
            }

            float cur = 0f;
            for (int i = 0; i < w; i++) cur += energy[i];

            float best = cur;
            int bestAt = 0;
            for (int i = 0; i + w < n; i++)
            {
                cur += energy[i + w] - energy[i];
                if (cur > best) { best = cur; bestAt = i + 1; }
            }

            return Sub(mix, bestAt * frame, (int)(seconds * VoxLockConfig.SampleRate));
        }

        /// <summary>
        /// 把声音样本变成网络要的形式：正好 3 s。样本更长时取前 30 s 里最响的 3 s。
        /// </summary>
        public static float[] FitSample(float[] sample)
        {
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            if (sample.Length < VoxLockConfig.SampleLength)
                throw new ArgumentException(
                    $"the voice sample must be at least 3 s long, got {sample.Length / (float)VoxLockConfig.SampleRate:F2} s");

            if (sample.Length == VoxLockConfig.SampleLength) return sample;

            float[] loud = LoudSample(sample, 3f, VoxLockConfig.LoudWithinLong / (float)VoxLockConfig.SampleRate);
            if (loud.Length == VoxLockConfig.SampleLength) return loud;
            return Sub(loud, 0, VoxLockConfig.SampleLength);
        }

        /// <summary>
        /// 整通电话进，说话人出，长度不变。不给 <paramref name="sample"/> 时从通话本身取
        /// <see cref="LoudSample"/>；短于 3 s 的通话没有样本可取，于是原样返回。
        /// </summary>
        public static float[] Isolate(VoxLockSession session, float[] audio, float[] sample = null,
                                      int framesPerRun = VoxLockConfig.DefaultFramesPerRun)
        {
            if (session == null) throw new ArgumentNullException(nameof(session));
            if (audio == null) throw new ArgumentNullException(nameof(audio));

            if (sample == null && audio.Length < VoxLockConfig.SampleLength) return Sub(audio, 0, audio.Length);

            using (var stream = CreateStream(session, sample ?? LoudSample(audio), framesPerRun))
            {
                // 输出第 i 个样本对应输入第 i 个样本——开头那 256 个填充样本已经被丢掉——所以流的输出可以
                // 直接写进一个与输入同下标的缓冲。
                var output = new float[audio.Length + VoxLockConfig.FftSize];
                var scratch = new float[VoxLockStream.OutputCapacityFor(IsolateChunk)];
                int written = 0;

                int at = 0;
                while (at < audio.Length)
                {
                    int n = Math.Min(IsolateChunk, audio.Length - at);
                    int produced = stream.Push(audio, at, n, scratch);
                    if (produced > 0)
                    {
                        Array.Copy(scratch, 0, output, written, produced);
                        written += produced;
                    }
                    at += n;
                }

                var tail = new float[stream.FlushCapacity];
                int tailCount = stream.Flush(tail);
                if (tailCount > 0)
                {
                    Array.Copy(tail, 0, output, written, tailCount);
                    written += tailCount;
                }

                int total = Math.Min(audio.Length, written);
                var result = new float[total];
                Array.Copy(output, result, total);
                return result;
            }
        }

        /// <summary>
        /// 从第一块就开始隔离，用于你已经有声音样本的场景——来自通话更早的部分，或者来自上一次通话。
        /// <paramref name="sample"/> 至少要 3 s；用 <see cref="FitSample"/> 处理。
        /// </summary>
        public static VoxLockStream CreateStream(VoxLockSession session, float[] sample,
                                                 int framesPerRun = VoxLockConfig.DefaultFramesPerRun)
        {
            if (session == null) throw new ArgumentNullException(nameof(session));
            return new VoxLockStream(session, FitSample(sample), framesPerRun);
        }

        /// <summary>
        /// 自己找出说话人的实时流：先原样透传，听够 <paramref name="seconds"/> 秒语音后把这些语音当作声音
        /// 样本，从那时起开始隔离。<paramref name="seconds"/> 至少为 3。
        /// </summary>
        public static VoxLockLiveStream CreateLive(VoxLockSession session, float seconds = 3f, float margin = 10f,
                                                   int framesPerRun = VoxLockConfig.DefaultFramesPerRun)
        {
            if (session == null) throw new ArgumentNullException(nameof(session));
            return new VoxLockLiveStream(session, seconds, margin, framesPerRun);
        }

        /// <summary>
        /// <see cref="Isolate"/> 每次交给流多少音频。大到能让模型保持忙碌，小到能约束暂存缓冲——整文件跑
        /// 的时候延迟无所谓。
        /// </summary>
        private const int IsolateChunk = 10 * VoxLockConfig.SampleRate;

        // ---- 文件处理 ------------------------------------------------------------------

        /// <summary>
        /// 整文件处理：读一个 WAV，整段隔离，写成 16 位单声道 16 kHz 的 WAV。
        /// </summary>
        /// <param name="session">加载好的模型。</param>
        /// <param name="inputPath">输入 WAV，16 位 PCM；非 16 kHz 会重采样，立体声会下混。</param>
        /// <param name="outputPath">输出 WAV，总是 16 kHz 单声道。</param>
        /// <param name="sample">可选的声音样本；不给就从输入里取最响的 3 s。</param>
        /// <returns>处理统计。</returns>
        public static VoxLockFileResult IsolateFile(VoxLockSession session, string inputPath, string outputPath,
                                                    float[] sample = null,
                                                    int framesPerRun = VoxLockConfig.DefaultFramesPerRun)
        {
            if (session == null) throw new ArgumentNullException(nameof(session));
            if (inputPath == null) throw new ArgumentNullException(nameof(inputPath));
            if (outputPath == null) throw new ArgumentNullException(nameof(outputPath));

            var clock = System.Diagnostics.Stopwatch.StartNew();

            float[] audio = VoxLockWav.Read(inputPath, out int sourceRate);
            int inputSamples = audio.Length;

            if (sourceRate != VoxLockConfig.SampleRate)
                audio = Resample(audio, sourceRate, VoxLockConfig.SampleRate);

            var result = new VoxLockFileResult
            {
                SourceSampleRate = sourceRate,
                InputSamples = inputSamples,
                InputSeconds = inputSamples / (double)sourceRate,
            };

            // 短于 3 s 没有样本可取，原样写出，与 Isolate 的行为保持一致。
            if (audio.Length < VoxLockConfig.SampleLength)
            {
                VoxLockWav.Write(outputPath, audio, VoxLockConfig.SampleRate);
                result.OutputSamples = audio.Length;
                result.Skipped = true;
                result.Seconds = clock.Elapsed.TotalSeconds;
                return result;
            }

            float[] isolated = Isolate(session, audio, sample, framesPerRun);
            VoxLockWav.Write(outputPath, isolated, VoxLockConfig.SampleRate);

            result.OutputSamples = isolated.Length;
            result.Seconds = clock.Elapsed.TotalSeconds;
            return result;
        }

        /// <summary>
        /// 流式文件处理：按 <paramref name="blockSamples"/> 个样本一块读、一块隔离、一块写盘。
        /// 内存占用与文件长度无关，适合长录音。
        ///
        /// 分两遍：第一遍只算每 10 ms 帧的能量，找出最响的 3 s 当声音样本；第二遍才真正跑模型。
        /// 两遍都是从磁盘按块读，所以即便是一小时的录音，峰值内存也只有几十 KB。
        ///
        /// 与 <see cref="IsolateFile"/> 的区别只在于数据怎么流动：管线本身是同一套流式实现，输出逐样本一致
        /// （int8 模型因量化会有 ~1e-3 量级的差别，见 README），输出长度也与输入一致。
        /// </summary>
        /// <param name="session">加载好的模型。</param>
        /// <param name="inputPath">输入 WAV。</param>
        /// <param name="outputPath">输出 WAV，总是 16 kHz 单声道。</param>
        /// <param name="blockSamples">每次喂给流的样本数，默认 20 ms。必须是正数。</param>
        /// <param name="sample">可选的声音样本；不给就自己从输入里找最响的 3 s。</param>
        /// <param name="onProgress">每处理完一块回调一次，参数是已处理秒数和总秒数；可以为 null。</param>
        public static VoxLockFileResult IsolateFileStreaming(VoxLockSession session, string inputPath,
                                                             string outputPath, int blockSamples = 320,
                                                             float[] sample = null,
                                                             Action<double, double> onProgress = null,
                                                             int framesPerRun = VoxLockConfig.DefaultFramesPerRun)
        {
            if (session == null) throw new ArgumentNullException(nameof(session));
            if (inputPath == null) throw new ArgumentNullException(nameof(inputPath));
            if (outputPath == null) throw new ArgumentNullException(nameof(outputPath));
            if (blockSamples < 1) throw new ArgumentOutOfRangeException(nameof(blockSamples));

            var clock = System.Diagnostics.Stopwatch.StartNew();

            var result = new VoxLockFileResult();

            using (var probe = new VoxLockWavReader(inputPath))
            {
                result.SourceSampleRate = probe.SourceSampleRate;
                result.InputSamples = (int)probe.SourceFrames;
                result.InputSeconds = probe.Seconds;

                if (probe.Seconds * VoxLockConfig.SampleRate < VoxLockConfig.SampleLength)
                {
                    // 短于 3 s：没有样本可取，原样转成 16 kHz 单声道写出。
                    CopyAsIs(inputPath, outputPath, out long copied);
                    result.OutputSamples = copied;
                    result.Skipped = true;
                    result.Seconds = clock.Elapsed.TotalSeconds;
                    return result;
                }
            }

            // 第一遍：定位声音样本。只需要样本本身进内存。
            float[] voice = sample != null
                ? FitSample(sample)
                : FitSample(FindLoudSample(inputPath));

            double totalSeconds = result.InputSeconds;

            // 第二遍：按块处理并写盘。
            using (var reader = new VoxLockWavReader(inputPath))
            using (var stream = new VoxLockStream(session, voice, framesPerRun))
            using (var writer = new VoxLockWavWriter(outputPath, VoxLockConfig.SampleRate))
            {
                var input = new float[blockSamples];
                var output = new float[VoxLockStream.OutputCapacityFor(blockSamples)];

                // 流的第 i 个输出样本对应输入的第 i 个样本，所以输出长度与输入一致。流在收尾时会把最后一个
                // 窗的窗尾也吐出来，那部分不对应任何输入样本，要丢掉——否则输出会比输入长几百个样本。
                long target = 0;

                while (true)
                {
                    int n = reader.Read(input);
                    if (n == 0) break;
                    target += n;

                    int produced = stream.Push(input, n, output);
                    WriteUpTo(writer, output, produced, target);

                    onProgress?.Invoke(writer.SecondsWritten, totalSeconds);
                }

                var tail = new float[stream.FlushCapacity];
                int tailCount = stream.Flush(tail);
                WriteUpTo(writer, tail, tailCount, target);

                result.OutputSamples = writer.SamplesWritten;
            }

            result.Seconds = clock.Elapsed.TotalSeconds;
            return result;
        }

        /// <summary>
        /// 把一块结果写进 <paramref name="writer"/>，但最多只写到第 <paramref name="target"/> 个样本——
        /// 那是输入的长度，也就是输出该有的长度。
        /// </summary>
        private static void WriteUpTo(VoxLockWavWriter writer, float[] data, int count, long target)
        {
            long room = target - writer.SamplesWritten;
            if (room <= 0) return;
            if (room < count) count = (int)room;
            writer.Write(data, 0, count);
        }

        /// <summary>
        /// 扫一遍文件，找出最响的 3 s 并读出来。对应 <see cref="LoudSample"/>，但不需要把整个文件读进内存。
        /// 只在前 15 s 里找，和评测口径一致。
        /// </summary>
        public static float[] FindLoudSample(string inputPath)
        {
            if (inputPath == null) throw new ArgumentNullException(nameof(inputPath));

            const int frame = VoxLockConfig.EnergyFrame;
            int windowFrames = VoxLockConfig.SampleLength / frame;          // 3 s 有多少个 10 ms 帧
            long withinFrames = VoxLockConfig.LoudWithin / frame;           // 只看前 15 s

            var energy = new System.Collections.Generic.List<float>();
            var buffer = new float[frame * 16];
            var leftover = new float[frame];
            int leftoverLen = 0;

            using (var reader = new VoxLockWavReader(inputPath))
            {
                while (energy.Count < withinFrames)
                {
                    int n = reader.Read(buffer);
                    if (n == 0) break;

                    int at = 0;

                    // 先补上上一块剩下的不足一帧的数据。
                    if (leftoverLen > 0)
                    {
                        int take = Math.Min(frame - leftoverLen, n);
                        Array.Copy(buffer, 0, leftover, leftoverLen, take);
                        leftoverLen += take;
                        at += take;

                        if (leftoverLen < frame)
                        {
                            // 这一块还不够凑满一帧，整块都进了 leftover，直接读下一块。
                            continue;
                        }

                        energy.Add(MeanSquare(leftover, 0, frame));
                        leftoverLen = 0;
                    }

                    while (at + frame <= n && energy.Count < withinFrames)
                    {
                        energy.Add(MeanSquare(buffer, at, frame));
                        at += frame;
                    }

                    if (at < n)
                    {
                        leftoverLen = n - at;
                        Array.Copy(buffer, at, leftover, 0, leftoverLen);
                    }
                }
            }

            if (energy.Count == 0) return new float[VoxLockConfig.SampleLength];

            int bestFrame = ArgMaxWindow(energy, windowFrames);
            long startSample = (long)bestFrame * frame;

            // 把那段读出来。需要 3 s，落在内存里只有 96 KB。
            var sample = new float[VoxLockConfig.SampleLength];
            using (var reader = new VoxLockWavReader(inputPath))
            {
                // 跳过样本起点之前的音频。用固定大小的缓冲循环丢弃，startSample 为 0 时循环不进。
                var discard = new float[4096];
                long skip = startSample;
                while (skip > 0)
                {
                    int want = (int)Math.Min(discard.Length, skip);
                    var slice = new float[want];
                    int n = reader.Read(slice);
                    if (n == 0) break;
                    skip -= n;
                }

                int filled = 0;
                while (filled < sample.Length)
                {
                    var slice = new float[sample.Length - filled];
                    int n = reader.Read(slice);
                    if (n == 0) break;
                    Array.Copy(slice, 0, sample, filled, n);
                    filled += n;
                }

                // 文件尾部不够 3 s 时补零，与 Sub() 的裁剪行为对齐。
                for (int i = filled; i < sample.Length; i++) sample[i] = 0f;
            }

            return sample;
        }

        private static float MeanSquare(float[] data, int offset, int count)
        {
            float sum = 0f;
            for (int i = 0; i < count; i++)
            {
                float v = data[offset + i];
                sum += v * v;
            }
            return sum / count;
        }

        /// <summary>滑动窗口和的最大值位置。能量数组太短时返回 0。</summary>
        private static int ArgMaxWindow(System.Collections.Generic.List<float> energy, int windowFrames)
        {
            int n = energy.Count;
            int w = Math.Min(windowFrames, n);
            if (w <= 0) return 0;

            float cur = 0f;
            for (int i = 0; i < w; i++) cur += energy[i];

            float best = cur;
            int bestAt = 0;
            for (int i = 0; i + w < n; i++)
            {
                cur += energy[i + w] - energy[i];
                if (cur > best) { best = cur; bestAt = i + 1; }
            }
            return bestAt;
        }

        /// <summary>不做隔离，只把文件转成 16 kHz 单声道写出去。</summary>
        private static void CopyAsIs(string inputPath, string outputPath, out long written)
        {
            using (var reader = new VoxLockWavReader(inputPath))
            using (var writer = new VoxLockWavWriter(outputPath, VoxLockConfig.SampleRate))
            {
                var buffer = new float[4096];
                while (true)
                {
                    int n = reader.Read(buffer);
                    if (n == 0) break;
                    writer.Write(buffer, 0, n);
                }
                written = writer.SamplesWritten;
            }
        }

        /// <summary>
        /// 线性重采样。模型只接受 16 kHz，输入文件可能是 44.1 或 48 kHz。
        /// 演示与预处理够用；要求高保真可以换成带限重采样。
        /// </summary>
        public static float[] Resample(float[] src, int from, int to)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (from <= 0 || to <= 0) throw new ArgumentOutOfRangeException(nameof(from));
            if (from == to) return src;

            int n = (int)(src.Length * (double)to / from);
            var dst = new float[n];
            double step = from / (double)to;

            for (int i = 0; i < n; i++)
            {
                double pos = i * step;
                int i0 = (int)pos;
                int i1 = Math.Min(i0 + 1, src.Length - 1);
                float frac = (float)(pos - i0);
                dst[i] = src[i0] + (src[i1] - src[i0]) * frac;
            }
            return dst;
        }

        private static float[] Sub(float[] src, int offset, int count)
        {
            var result = new float[count];
            Array.Copy(src, offset, result, 0, count);
            return result;
        }
    }

    /// <summary>一次文件处理的结果，用来打日志或显示进度。</summary>
    public struct VoxLockFileResult
    {
        /// <summary>输入文件原本的采样率。</summary>
        public int SourceSampleRate;
        /// <summary>输入文件的采样点数（重采样之前）。</summary>
        public int InputSamples;
        /// <summary>输入文件的时长，秒。</summary>
        public double InputSeconds;
        /// <summary>输出文件的采样点数（16 kHz）。</summary>
        public long OutputSamples;
        /// <summary>处理耗时，秒。</summary>
        public double Seconds;
        /// <summary>输入短于 3 s，没有做隔离，原样写出。</summary>
        public bool Skipped;

        /// <summary>实时因子：处理耗时 / 音频时长。小于 1 表示比实时快。</summary>
        public double RealTimeFactor => InputSeconds > 0 ? Seconds / InputSeconds : 0;

        public override string ToString()
        {
            if (Skipped) return $"skipped, shorter than 3 s ({InputSeconds:F2} s)";
            return $"{InputSeconds:F2} s in {Seconds:F2} s (RTF {RealTimeFactor:F2}), " +
                   $"{OutputSamples} samples out at {VoxLockConfig.SampleRate} Hz";
        }
    }
}
