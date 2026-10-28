// VoxLock - 锁定到某个声音样本的流式隔离。
//
// 管线与参考实现逐样本一致：
//   1. 输入两端各做 256 个样本的反射填充（torch.stft 的中心填充）
//   2. 每次在 (k-1)*128 + 512 个样本上跑 chunk 图，并携带 LSTM 状态
//   3. 用 Hann^2 窗把 512 点帧做重叠相加，再除以累加窗
//   4. 丢掉开头 256 个填充样本，然后乘上 GAIN
//
// 唯一的结构性差异是重叠相加放在固定的环形缓冲里，而不是一个每帧都要增长并拷贝的数组。参考实现每帧都要
// 重新分配整个尾部，长通话下是平方复杂度；这里是线性。
//
// 输出大小：流追赶积压时，一次最多能吐出比你推入的多 512 个样本。输出缓冲请用
// OutputCapacityFor(count) 来分配。
using System;
using System.Collections.Generic;

namespace VoxLock
{
    /// <summary>
    /// 锁定到构造时给定样本中那个声音的实时流。音频按到达顺序推入，块大小任意；每次推入返回已经就绪的
    /// 隔离音频，大约滞后 24 ms。
    /// </summary>
    public sealed class VoxLockStream : IDisposable
    {
        private const int CompactThreshold = 4096;

        private readonly VoxLockSession _session;
        private readonly VoiceEmbedding _voice;
        private readonly int _framesPerRun;
        private readonly float[] _frames;

        private float[] _h;
        private float[] _c;

        private readonly List<float> _head = new List<float>();
        private float[] _buf = Array.Empty<float>();
        private int _bufStart;
        private int _bufLen;
        private bool _hasBuf;

        // 重叠相加的环形缓冲。活跃窗口长度恒为 NFFT；`_olaPos` 是它的起点。
        // 已经吐出去的位置会被清零，这正是下一次累加能保持正确、而无需清空其它位置的原因。容量足够大，
        // 保证活跃窗口与正在吐出的样本永不重叠，所以取模运算没有歧义。
        private readonly float[] _ola;
        private readonly float[] _env;
        private readonly int _olaCap;
        private int _olaPos;

        private int _dropped;
        private int _nIn;
        private bool _flushed;

        public VoxLockStream(VoxLockSession session, float[] sample,
                             int framesPerRun = VoxLockConfig.DefaultFramesPerRun)
        {
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            if (sample.Length < VoxLockConfig.SampleLength)
                throw new ArgumentException(
                    $"the voice sample must be at least 3 s long, got {sample.Length / (float)VoxLockConfig.SampleRate:F2} s");
            if (framesPerRun < 1) throw new ArgumentOutOfRangeException(nameof(framesPerRun));

            _session = session ?? throw new ArgumentNullException(nameof(session));
            _framesPerRun = framesPerRun;
            _frames = new float[framesPerRun * VoxLockConfig.FftSize];

            _voice = session.EnrollSample(sample);
            _h = new float[session.StateLength];
            _c = new float[session.StateLength];

            _olaCap = VoxLockConfig.FftSize + framesPerRun * VoxLockConfig.Hop;
            _ola = new float[_olaCap];
            _env = new float[_olaCap];
        }

        /// <summary>
        /// 推入 <paramref name="count"/> 个样本需要多大的输出缓冲。流会压住不到一个窗口的数据，所以追赶
        /// 时最多能吐出比推入多这么多的样本。
        /// </summary>
        public static int OutputCapacityFor(int count) => count + VoxLockConfig.FftSize;

        /// <summary>到目前为止推入的样本数，以及管线还压着的样本数。</summary>
        public int SamplesIn => _nIn;
        public int PendingInput => _hasBuf ? _bufLen : 0;

        /// <summary><see cref="Flush"/> 是否已被调用。</summary>
        public bool Flushed => _flushed;

        /// <summary>
        /// 推入任意数量的 16 kHz 样本；已经就绪的隔离音频从 <paramref name="output"/> 输出，比输入滞后
        /// 24 ms。<paramref name="output"/> 至少要能容纳 <paramref name="count"/> 对应的
        /// <see cref="OutputCapacityFor"/>。
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
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (output.Length < OutputCapacityFor(count))
                throw new ArgumentException(
                    $"output must hold at least {OutputCapacityFor(count)} samples for a push of {count}", nameof(output));
            if (_flushed) throw new InvalidOperationException("the stream has been flushed");

            _nIn += count;
            if (count == 0) return 0;

            if (!_hasBuf)
            {
                // 还不够构造开头填充的数据。
                for (int i = 0; i < count; i++) _head.Add(chunk[offset + i]);
                if (_head.Count <= VoxLockConfig.Pad) return 0;

                // torch.stft 的中心填充：把开头 256 个样本反射到音频前面。
                var b = new float[VoxLockConfig.Pad + _head.Count];
                for (int i = 0; i < VoxLockConfig.Pad; i++) b[i] = _head[VoxLockConfig.Pad - i];
                for (int i = 0; i < _head.Count; i++) b[VoxLockConfig.Pad + i] = _head[i];

                _buf = b;
                _bufStart = 0;
                _bufLen = b.Length;
                _dropped = VoxLockConfig.Pad;
                _head.Clear();
                _hasBuf = true;
            }
            else
            {
                EnsureRoom(count);
                Array.Copy(chunk, offset, _buf, _bufStart + _bufLen, count);
                _bufLen += count;
            }

            return Run(output);
        }

        /// <summary>会自己分配输出的便捷重载。循环里请用带暂存缓冲的那个。</summary>
        public float[] Push(float[] chunk, int count)
        {
            var output = new float[OutputCapacityFor(count)];
            int n = Push(chunk, count, output);
            if (n == output.Length) return output;
            var trimmed = new float[n];
            Array.Copy(output, trimmed, n);
            return trimmed;
        }

        /// <summary>
        /// 通话结束：返回最后还压着的样本。之后不能再推入。缓冲区大小用 <see cref="FlushCapacity"/>。
        /// </summary>
        public int Flush(float[] output)
        {
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (_flushed) return 0;
            _flushed = true;

            if (!_hasBuf) return 0;

            int need = FlushCapacity;
            if (output.Length < need)
                throw new ArgumentException($"output must hold at least {need} samples", nameof(output));

            // 尾部填充：把最后 256 个样本（不含最后一个）反射到音频后面。
            int n = _bufLen;
            int tail = Math.Max(0, Math.Min(VoxLockConfig.Pad, n - 1));

            EnsureRoom(tail);
            for (int i = 0; i < tail; i++)
                _buf[_bufStart + n + i] = _buf[_bufStart + n - 2 - i];

            int xLen = n + tail;
            _bufLen = xLen;

            // 用零补齐到整数帧。
            int nfr = 1 + (xLen - VoxLockConfig.FftSize) / VoxLockConfig.Hop;
            if (nfr < 1) nfr = 1;
            int want = (nfr - 1) * VoxLockConfig.Hop + VoxLockConfig.FftSize;
            if (want > xLen)
            {
                EnsureRoom(want - xLen);
                for (int i = _bufStart + xLen; i < _bufStart + want; i++) _buf[i] = 0f;
                _bufLen = want;
            }

            int produced = Run(output);

            // 重叠相加窗口剩下的部分。
            for (int i = 0; i < VoxLockConfig.FftSize && produced < output.Length; i++)
            {
                int idx = (_olaPos + i) % _olaCap;
                float e = _env[idx];
                output[produced++] = (e > 1e-11f ? _ola[idx] / e : 0f) * VoxLockConfig.Gain;
            }
            return produced;
        }

        /// <summary><see cref="Flush"/> 需要多大的缓冲：压着的输入，再加一个窗口。</summary>
        public int FlushCapacity => _hasBuf ? _bufLen + 2 * VoxLockConfig.FftSize : VoxLockConfig.FftSize;

        private int Run(float[] output)
        {
            int produced = 0;

            while (_hasBuf && _bufLen >= VoxLockConfig.FftSize)
            {
                int maxFrames = 1 + (_bufLen - VoxLockConfig.FftSize) / VoxLockConfig.Hop;
                int k = Math.Min(_framesPerRun, maxFrames);
                int segLen = (k - 1) * VoxLockConfig.Hop + VoxLockConfig.FftSize;

                _session.RunChunk(_buf, _bufStart, segLen, _voice, _h, _c, _frames);

                for (int j = 0; j < k; j++)
                {
                    int hopOffset = j * VoxLockConfig.Hop;
                    int frameBase = j * VoxLockConfig.FftSize;
                    for (int i = 0; i < VoxLockConfig.FftSize; i++)
                    {
                        int idx = (_olaPos + hopOffset + i) % _olaCap;
                        _ola[idx] += _frames[frameBase + i];
                        _env[idx] += VoxLockConfig.WindowSquared[i];
                    }
                }

                int done = k * VoxLockConfig.Hop;
                int blockStart = produced;

                for (int t = 0; t < done; t++)
                {
                    int idx = (_olaPos + t) % _olaCap;
                    float e = _env[idx];
                    output[produced++] = (e > 1e-11f ? _ola[idx] / e : 0f) * VoxLockConfig.Gain;
                    _ola[idx] = 0f;
                    _env[idx] = 0f;
                }
                _olaPos = (_olaPos + done) % _olaCap;

                // 流开头那 256 个填充样本不属于通话内容。
                int cut = Math.Min(_dropped, produced - blockStart);
                if (cut > 0)
                {
                    _dropped -= cut;
                    Array.Copy(output, blockStart + cut, output, blockStart, produced - blockStart - cut);
                    produced -= cut;
                }

                _bufStart += done;
                _bufLen -= done;
                Compact();
            }

            return produced;
        }

        /// <summary>
        /// 为已经缓冲的样本之后再腾出 <paramref name="extra"/> 个样本的空间。缓冲区从 <c>_bufStart</c>
        /// 开始使用，所以实际需要的空间是 <c>_bufStart + _bufLen + extra</c>；只判断
        /// <c>_bufLen + extra</c> 的话，起点一旦前移就会写到末尾之外。
        /// </summary>
        private void EnsureRoom(int extra)
        {
            int needed = _bufStart + _bufLen + extra;
            if (needed <= _buf.Length) return;

            // 先把活跃样本滑到最前面——这一步不花钱，通常就够了。
            if (_bufLen + extra <= _buf.Length)
            {
                if (_bufLen > 0) Array.Copy(_buf, _bufStart, _buf, 0, _bufLen);
                _bufStart = 0;
                return;
            }

            int newSize = Math.Max(1024, _buf.Length);
            while (newSize < _bufLen + extra) newSize *= 2;

            var grown = new float[newSize];
            if (_bufLen > 0) Array.Copy(_buf, _bufStart, grown, 0, _bufLen);
            _buf = grown;
            _bufStart = 0;
        }

        private void Compact()
        {
            if (_bufStart < CompactThreshold) return;
            if (_bufLen > 0) Array.Copy(_buf, _bufStart, _buf, 0, _bufLen);
            _bufStart = 0;
        }

        public void Dispose()
        {
            _voice?.Dispose();
        }
    }
}
