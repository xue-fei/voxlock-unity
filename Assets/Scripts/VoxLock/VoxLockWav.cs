// VoxLock - WAV 读写。
//
// Python 那边输入输出都是 numpy 数组，Unity 这边得自己处理文件，所以 RIFF 的解析和写入放在这里，
// 库和 Demo 共用。
//
// 读只支持 16 位 PCM（单声道，或立体声下混为单声道），采样率任意。写固定 16 位单声道——模型工作在
// 16 kHz 单声道 float32，格式转换只在边界发生一次。
//
// VoxLockWavWriter 支持流式写入：先写 44 字节占位头，每处理完一块就落盘，关闭时回填 RIFF 和 data 的
// 长度。这样流式处理不需要把整段结果攒在内存里，处理多少就写多少。
using System;
using System.IO;
using System.Text;

namespace VoxLock
{
    public static class VoxLockWav
    {
        /// <summary>标准 PCM WAV 头长度。</summary>
        public const int HeaderSize = 44;

        /// <summary>
        /// 读一个 16 位 PCM WAV。立体声会下混成单声道，采样率从文件里读出来。
        /// </summary>
        public static float[] Read(string path, out int sampleRate)
        {
            if (path == null) throw new ArgumentNullException(nameof(path));
            return Read(File.ReadAllBytes(path), out sampleRate);
        }

        /// <summary>同上，直接从内存里的字节读。</summary>
        public static float[] Read(byte[] bytes, out int sampleRate)
        {
            if (bytes == null) throw new ArgumentNullException(nameof(bytes));
            if (bytes.Length < HeaderSize || Encoding.ASCII.GetString(bytes, 0, 4) != "RIFF")
                throw new InvalidDataException("not a RIFF file");

            int channels = 1, bits = 16, at = 12, dataAt = -1, dataLen = 0;
            sampleRate = VoxLockConfig.SampleRate;

            while (at + 8 <= bytes.Length)
            {
                string id = Encoding.ASCII.GetString(bytes, at, 4);
                int size = BitConverter.ToInt32(bytes, at + 4);
                int body = at + 8;

                if (id == "fmt ")
                {
                    channels = BitConverter.ToInt16(bytes, body + 2);
                    sampleRate = BitConverter.ToInt32(bytes, body + 4);
                    bits = BitConverter.ToInt16(bytes, body + 14);
                }
                else if (id == "data")
                {
                    dataAt = body;
                    dataLen = Math.Min(size, bytes.Length - body);
                    break;
                }

                // 块大小是奇数时后面有一个填充字节。
                at = body + size + (size & 1);
            }

            if (dataAt < 0) throw new InvalidDataException("no data chunk");
            if (channels < 1) throw new InvalidDataException($"bad channel count {channels}");
            if (bits != 16) throw new InvalidDataException($"only 16-bit PCM is read here, got {bits}-bit");

            int frames = dataLen / (2 * channels);
            var audio = new float[frames];
            for (int i = 0; i < frames; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                    sum += BitConverter.ToInt16(bytes, dataAt + (i * channels + c) * 2) / 32768f;
                audio[i] = sum / channels;
            }
            return audio;
        }

        /// <summary>把整段音频写成 16 位单声道 WAV。</summary>
        public static void Write(string path, float[] data, int sampleRate)
        {
            using (var writer = new VoxLockWavWriter(path, sampleRate))
                writer.Write(data, 0, data == null ? 0 : data.Length);
        }
    }

    /// <summary>
    /// 流式 WAV 读取器：按块读出 16 kHz 单声道 float，内存占用与文件长度无关。
    ///
    /// 下混和重采样都在这里做掉，调用方拿到的永远是模型要的格式，可以一块一块喂给
    /// <see cref="VoxLockLiveStream"/> 或 <see cref="VoxLockStream"/>。重采样是有状态的——小数读取位置和
    /// 上一个采样跨块保留——否则每个块边界都会引入一个断点。
    /// </summary>
    public sealed class VoxLockWavReader : IDisposable
    {
        private readonly FileStream _stream;
        private readonly int _sourceRate;
        private readonly int _channels;
        private readonly long _frames;
        private readonly int _blockFrames;

        private readonly float[] _source;
        private readonly byte[] _bytes;
        private readonly bool _resample;

        // 待取队列：磁盘读一块、重采样后先攒在这里，调用方按自己的节奏取。
        private readonly float[] _pending;
        private int _pendingAt;
        private int _pendingLen;

        private long _frameAt;
        private double _resamplePos;
        private float _previous;
        private bool _disposed;

        /// <summary>文件原本的采样率。</summary>
        public int SourceSampleRate => _sourceRate;

        /// <summary>文件里的帧数（重采样之前的采样点数）。</summary>
        public long SourceFrames => _frames;

        /// <summary>音频时长，秒。</summary>
        public double Seconds => _frames / (double)_sourceRate;

        /// <summary>已经读出的 16 kHz 采样点数。</summary>
        public long SamplesRead { get; private set; }

        /// <summary>是否读到了末尾。</summary>
        public bool EndOfFile => _frameAt >= _frames;

        /// <param name="path">16 位 PCM WAV。</param>
        /// <param name="blockFrames">每次从磁盘读多少源帧，默认 4096。</param>
        public VoxLockWavReader(string path, int blockFrames = 4096)
        {
            if (path == null) throw new ArgumentNullException(nameof(path));
            if (blockFrames < 64) throw new ArgumentOutOfRangeException(nameof(blockFrames));

            _stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            _blockFrames = blockFrames;

            // 只解析头部，然后停 data 块的起点——后面按需 seek 读。
            var header = new byte[Math.Min(4096, (int)_stream.Length)];
            _stream.Read(header, 0, header.Length);

            int at = 12, dataAt = -1, dataLen = 0, channels = 1, bits = 16;
            _sourceRate = VoxLockConfig.SampleRate;

            while (at + 8 <= header.Length)
            {
                string id = Encoding.ASCII.GetString(header, at, 4);
                int size = BitConverter.ToInt32(header, at + 4);
                int body = at + 8;

                if (id == "fmt ")
                {
                    channels = BitConverter.ToInt16(header, body + 2);
                    _sourceRate = BitConverter.ToInt32(header, body + 4);
                    bits = BitConverter.ToInt16(header, body + 14);
                }
                else if (id == "data")
                {
                    dataAt = body;
                    dataLen = size;
                    break;
                }
                at = body + size + (size & 1);
            }

            if (dataAt < 0) throw new InvalidDataException($"{path} has no data chunk");
            if (channels < 1) throw new InvalidDataException($"bad channel count {channels}");
            if (bits != 16) throw new InvalidDataException($"only 16-bit PCM is read here, got {bits}-bit");

            // data 块声明的长度可能超过文件实际长度（被截断的文件），取小的那个。
            long available = Math.Max(0, _stream.Length - dataAt);
            if (dataLen > available) dataLen = (int)available;

            _channels = channels;
            _frames = dataLen / (2 * channels);
            _stream.Position = dataAt;

            _source = new float[_blockFrames];
            _bytes = new byte[_blockFrames * 2 * _channels];

            // 重采样后数据量可能略多于源块（上采样时），留一点余量。
            int pendingCap = _blockFrames + 16;
            _pending = new float[pendingCap];

            _resample = _sourceRate != VoxLockConfig.SampleRate;
        }

        /// <summary>
        /// 读下一块 16 kHz 单声道样本，返回采样点数；读到末尾返回 0。
        ///
        /// 内部有一个待取队列：磁盘按 <see cref="_blockFrames"/> 读一次，重采样后的数据先攒着，调用方要
        /// 多少给多少。所以 destination 比 _blockFrames 小也没关系，不会丢数据。
        /// </summary>
        public int Read(float[] destination)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(VoxLockWavReader));
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (destination.Length == 0) return 0;

            int written = 0;
            while (written < destination.Length)
            {
                // 待取队列空了就再从磁盘拉一块。
                if (_pendingAt >= _pendingLen)
                {
                    if (!FillPending()) break;
                }

                int take = Math.Min(destination.Length - written, _pendingLen - _pendingAt);
                Array.Copy(_pending, _pendingAt, destination, written, take);
                _pendingAt += take;
                written += take;
            }

            SamplesRead += written;
            return written;
        }

        /// <summary>
        /// 从磁盘读一块并转换到 <see cref="_pending"/>。读到末尾返回 false。
        /// </summary>
        private bool FillPending()
        {
            if (_frameAt >= _frames) return false;

            int want = (int)Math.Min(_blockFrames, _frames - _frameAt);
            int bytes = want * 2 * _channels;

            // Read 可能返回不足请求的字节数，循环到读满或者到文件尾。
            int got = 0;
            while (got < bytes)
            {
                int n = _stream.Read(_bytes, got, bytes - got);
                if (n <= 0) break;
                got += n;
            }
            want = got / (2 * _channels);
            if (want <= 0) { _frameAt = _frames; return false; }

            for (int i = 0; i < want; i++)
            {
                float sum = 0f;
                for (int c = 0; c < _channels; c++)
                    sum += BitConverter.ToInt16(_bytes, (i * _channels + c) * 2) / 32768f;
                _source[i] = sum / _channels;
            }
            _frameAt += want;

            _pendingAt = 0;

            if (!_resample)
            {
                Array.Copy(_source, _pending, want);
                _pendingLen = want;
                return _pendingLen > 0;
            }

            // 线性重采样。小数位置和上一个采样跨块保留，块边界才不会出现断点。
            double ratio = _sourceRate / (double)VoxLockConfig.SampleRate;
            int produced = 0;

            while (produced < _pending.Length)
            {
                int i0 = (int)_resamplePos;
                // 需要 i0 和 i0+1 都在本块内；最后一点留给下一块插值。
                if (i0 + 1 >= want) break;

                float a = i0 < 0 ? _previous : _source[i0];
                float b = _source[i0 + 1];
                _pending[produced++] = a + (b - a) * (float)(_resamplePos - i0);
                _resamplePos += ratio;
            }

            _resamplePos -= want;
            _previous = _source[want - 1];
            _pendingLen = produced;
            return _pendingLen > 0;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _stream?.Dispose();
        }
    }

    /// <summary>
    /// 流式 WAV 写入器：打开时写占位头，之后每块追加，关闭时回填长度。
    /// 处理一块写一块，内存占用与音频长度无关。
    /// </summary>
    public sealed class VoxLockWavWriter : IDisposable
    {
        private readonly FileStream _stream;
        private readonly BinaryWriter _writer;
        private readonly int _sampleRate;
        private readonly int _channels;

        private long _dataBytes;
        private bool _closed;

        public VoxLockWavWriter(string path, int sampleRate, int channels = 1)
        {
            if (path == null) throw new ArgumentNullException(nameof(path));
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));

            _sampleRate = sampleRate;
            _channels = channels;

            // 目标目录可能还不存在（比如默认的 Assets/Isolated/）。这里建出来，调用方就不用自己操心。
            string dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);

            _stream = new FileStream(path, FileMode.Create, FileAccess.Write);
            _writer = new BinaryWriter(_stream);
            WriteHeader(0);
        }

        /// <summary>已经写出的帧数（每帧一个采样点，单声道）。</summary>
        public long SamplesWritten => _dataBytes / (2 * _channels);

        /// <summary>已经写出的音频秒数。</summary>
        public double SecondsWritten => SamplesWritten / (double)_sampleRate;

        public void Write(float[] data, int count) => Write(data, 0, count);

        /// <summary>追加一块音频。不足一帧的尾巴会被丢掉——WAV 的数据长度必须是帧的整数倍。</summary>
        public void Write(float[] data, int offset, int count)
        {
            if (_closed) throw new ObjectDisposedException(nameof(VoxLockWavWriter));
            if (data == null || count <= 0) return;
            if (offset < 0 || offset + count > data.Length)
                throw new ArgumentOutOfRangeException(nameof(count));

            int samples = count - count % _channels;
            for (int i = 0; i < samples; i++)
            {
                float v = data[offset + i];
                if (v > 1f) v = 1f;
                else if (v < -1f) v = -1f;
                _writer.Write((short)(v * 32767f));
            }
            _dataBytes += samples * 2;
        }

        public void Dispose()
        {
            if (_closed) return;
            _closed = true;

            // 回到开头重写头，把真实长度填进去。数据部分原地不动。
            _stream.Seek(0, SeekOrigin.Begin);
            WriteHeader(_dataBytes);
            _writer.Flush();
            _writer.Dispose();
        }

        private void WriteHeader(long dataBytes)
        {
            int blockAlign = _channels * 2;
            _writer.Write(Encoding.ASCII.GetBytes("RIFF"));
            _writer.Write((int)(36 + dataBytes));
            _writer.Write(Encoding.ASCII.GetBytes("WAVE"));
            _writer.Write(Encoding.ASCII.GetBytes("fmt "));
            _writer.Write(16);                    // fmt 块大小
            _writer.Write((short)1);              // 1 = PCM
            _writer.Write((short)_channels);
            _writer.Write(_sampleRate);
            _writer.Write(_sampleRate * blockAlign);   // 字节率
            _writer.Write((short)blockAlign);
            _writer.Write((short)16);             // 位深
            _writer.Write(Encoding.ASCII.GetBytes("data"));
            _writer.Write((int)dataBytes);
        }
    }
}
