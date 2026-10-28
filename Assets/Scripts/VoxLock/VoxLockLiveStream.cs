// VoxLock - 自己找出说话人的实时流。
//
// 在听够 3 s 语音之前，音频原样透传；这些语音就成为声音样本，从那时起开始隔离。"语音"的定义是能量比噪声
// 底高出至少 `margin` dB 的 10 ms 帧，噪声底取最近 2 s 里最安静的一帧——窗口最小值，所以不停顿地连续说话
// 的人也能凑满样本。数字静音（低于 -90 dBFS，比如麦克风刚打开之前）不计入噪声底，否则静音之后的线路噪声
// 就会被当成语音。
//
// 切换的那一刻，先把流在最近 250 ms 的音频上跑一遍并丢掉输出，这样它就不是从空的 LSTM 状态起步；同时隔离
// 音频在 10 ms 内淡入。
using System;
using System.Collections.Generic;

namespace VoxLock
{
    public sealed class VoxLockLiveStream : IDisposable
    {
        private const int Warm = 4000;    // 最近 250 ms，在切换前先过一遍模型
        private const int Fade = 160;     // 10 ms 淡入
        private const int FloorFrames = 200;   // 2 s，由 10 ms 帧组成，用来算噪声底
        private const int FloorWarmup = 20;    // 至少要有 20 帧，噪声底才有意义

        private readonly VoxLockSession _session;
        private readonly int _need;
        private readonly float _margin;
        private readonly int _framesPerRun;

        private readonly List<float> _voiced = new List<float>();
        private readonly Queue<float> _energy = new Queue<float>(FloorFrames);
        private float[] _rest = Array.Empty<float>();
        private float[] _recent = Array.Empty<float>();

        private VoxLockStream _stream;
        private float[] _scratch = Array.Empty<float>();

        private int _skip;
        private float[] _after = Array.Empty<float>();
        private int _afterLen;
        private int _faded;

        public VoxLockLiveStream(VoxLockSession session, float seconds = 3f, float margin = 10f,
                                 int framesPerRun = VoxLockConfig.DefaultFramesPerRun)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            if (seconds < VoxLockConfig.SampleLength / (float)VoxLockConfig.SampleRate)
                throw new ArgumentOutOfRangeException(nameof(seconds), $"seconds must be at least 3, got {seconds}");
            _need = (int)(seconds * VoxLockConfig.SampleRate / VoxLockConfig.EnergyFrame);
            _margin = margin;
            _framesPerRun = framesPerRun;
        }

        /// <summary>说话人是否已经找到、隔离是否已经开始。</summary>
        public bool Locked => _stream != null;

        /// <summary>到目前为止收集到的语音，以 10 ms 帧计。</summary>
        public int VoicedFrames => _voiced.Count / VoxLockConfig.EnergyFrame;

        /// <summary>距离锁定还差多少帧语音。</summary>
        public int FramesNeeded => Math.Max(0, _need - VoicedFrames);

        /// <summary>
        /// 推入 <paramref name="count"/> 个样本需要多大的输出缓冲。上限与
        /// <see cref="VoxLockStream.OutputCapacityFor"/> 相同：隔离路径追赶时最多能吐出比推入多一个窗口。
        /// </summary>
        public static int OutputCapacityFor(int count) => VoxLockStream.OutputCapacityFor(count);

        /// <summary>
        /// 推入音频。找到说话人之前原样透传；之后返回的隔离音频滞后约 24 ms。<paramref name="output"/>
        /// 至少要能容纳 <paramref name="count"/> 对应的 <see cref="OutputCapacityFor"/>。
        /// </summary>
        public int Push(float[] chunk, int count, float[] output)
        {
            return Push(chunk, 0, count, output);
        }

        /// <summary>带偏移的重载，调用方可以直接从一个大缓冲里推。</summary>
        public int Push(float[] chunk, int offset, int count, float[] output)
        {
            if (chunk == null) throw new ArgumentNullException(nameof(chunk));
            if (count < 0 || offset < 0 || offset + count > chunk.Length)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (output == null || output.Length < OutputCapacityFor(count))
                throw new ArgumentException(
                    $"output must hold at least {OutputCapacityFor(count)} samples for a push of {count}", nameof(output));

            if (_stream != null) return Isolated(chunk, offset, count, output);

            for (int i = 0; i < count; i++) output[i] = chunk[offset + i];

            _recent = AppendTail(_recent, chunk, offset, count);

            var x = Append(_rest, chunk, offset, count);
            int frames = x.Length / VoxLockConfig.EnergyFrame;
            int at = 0;
            for (int f = 0; f < frames; f++, at += VoxLockConfig.EnergyFrame)
            {
                float sum = 0f;
                for (int i = 0; i < VoxLockConfig.EnergyFrame; i++)
                {
                    float v = x[at + i];
                    sum += v * v;
                }
                float e = 10f * (float)Math.Log10(sum / VoxLockConfig.EnergyFrame + 1e-10f);

                if (e > -90f)
                {
                    if (_energy.Count >= FloorFrames) _energy.Dequeue();
                    _energy.Enqueue(e);
                }

                if (_energy.Count >= FloorWarmup && e >= Floor() + _margin)
                {
                    for (int i = 0; i < VoxLockConfig.EnergyFrame; i++) _voiced.Add(x[at + i]);
                }
            }
            _rest = Trim(x, frames * VoxLockConfig.EnergyFrame);

            if (_voiced.Count / VoxLockConfig.EnergyFrame >= _need) LockOn();
            return count;
        }

        /// <summary>通话结束：返回最后还压着的样本。</summary>
        public int Flush(float[] output)
        {
            return _stream?.Flush(output) ?? 0;
        }

        private void LockOn()
        {
            var sample = new float[_need * VoxLockConfig.EnergyFrame];
            for (int i = 0; i < sample.Length; i++) sample[i] = _voiced[i];

            _stream = new VoxLockStream(_session, sample, _framesPerRun);
            _voiced.Clear();
            _energy.Clear();

            _skip = _recent.Length;
            IsolatedWarm(_recent, _recent.Length);
        }

        private int Isolated(float[] chunk, int offset, int count, float[] output)
        {
            if (_faded < Fade)
            {
                _after = Append(_after, chunk, offset, count);
                _afterLen = _after.Length;
            }

            int produced = _stream.Push(chunk, offset, count, output);

            int cut = Math.Min(_skip, produced);
            _skip -= cut;
            if (cut > 0)
            {
                Array.Copy(output, cut, output, 0, produced - cut);
                produced -= cut;
            }

            if (_faded < Fade && produced > 0)
            {
                int n = Math.Min(produced, Fade - _faded);
                for (int i = 0; i < n; i++)
                {
                    int at = _faded + i;
                    float r = (at + 1) / (float)Fade;
                    float dry = at < _afterLen ? _after[at] : 0f;
                    output[i] = r * output[i] + (1f - r) * dry;
                }
                _faded += n;
                if (_faded >= Fade) { _after = Array.Empty<float>(); _afterLen = 0; }
            }

            return produced;
        }

        private void IsolatedWarm(float[] warm, int count)
        {
            if (_scratch.Length < VoxLockStream.OutputCapacityFor(count))
                Array.Resize(ref _scratch, VoxLockStream.OutputCapacityFor(count));
            int produced = _stream.Push(warm, count, _scratch);
            int cut = Math.Min(_skip, produced);
            _skip -= cut;
        }

        private float Floor()
        {
            float min = float.MaxValue;
            foreach (float e in _energy) if (e < min) min = e;
            return min;
        }

        private static float[] Append(float[] dst, float[] src, int offset, int count)
        {
            var result = new float[dst.Length + count];
            if (dst.Length > 0) Array.Copy(dst, result, dst.Length);
            Array.Copy(src, offset, result, dst.Length, count);
            return result;
        }

        private static float[] Trim(float[] src, int used)
        {
            int rest = src.Length - used;
            if (rest <= 0) return Array.Empty<float>();
            var result = new float[rest];
            Array.Copy(src, used, result, 0, rest);
            return result;
        }

        private static float[] AppendTail(float[] dst, float[] src, int offset, int count)
        {
            // dst ++ src 的最后 Warm 个样本。有上界，因为 dst 本身永远不会超过 Warm。
            var all = new float[dst.Length + count];
            if (dst.Length > 0) Array.Copy(dst, all, dst.Length);
            Array.Copy(src, offset, all, dst.Length, count);

            int keep = Math.Min(Warm, all.Length);
            if (keep == all.Length) return all;

            var tail = new float[keep];
            Array.Copy(all, all.Length - keep, tail, 0, keep);
            return tail;
        }

        public void Dispose()
        {
            _stream?.Dispose();
        }
    }
}
