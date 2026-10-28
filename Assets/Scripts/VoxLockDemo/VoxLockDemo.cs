// Demo 场景唯一的脚本：选个文件，选个模式，按按钮，看结果。
//
//   File      - 整文件处理。一次读入整段音频，整段隔离，写成一个 WAV。简单，但内存占用随文件长度增长。
//   Streaming - 流式处理。按 20 ms 一块读、一块隔离、一块落盘，内存占用与文件长度无关，适合长录音。
//               两种模式走的是同一套流式管线，输出逐样本一致，区别只在数据怎么流动。
//
// 没有播放，也不采集麦克风：这个 Demo 只负责把文件处理成文件。结果默认写进 Assets/Isolated/，
// 文件名是 <原名>_isolated.wav——放在 Assets 下，Unity 会把它导成 AudioClip，在 Project 窗口里点一下
// 就能试听，和输入音频一样方便。
//
// 模型运行放在工作线程上，所以处理长文件时编辑器不会卡住，进度行会实时更新。
using System;
using System.Collections;
using System.IO;
using System.Threading;
using UnityEngine;

namespace VoxLock
{
    public sealed class VoxLockDemo : MonoBehaviour
    {
        public enum Mode
        {
            /// <summary>整文件读入、整段隔离、一次写出。</summary>
            File = 0,
            /// <summary>按块流式处理，内存与文件长度无关。</summary>
            Streaming = 1,
        }

        [Header("Model")]
        [Tooltip("Which pair of ONNX files in StreamingAssets to load.")]
        public VoxLockPrecision Precision = VoxLockPrecision.Int8;

        [Tooltip("onnxruntime intra-op threads. 0 lets onnxruntime choose; 2 is a good default.")]
        [Range(0, 8)] public int Threads = 2;

        [Header("Input")]
        [Tooltip("File mode: absolute path, or a path under Assets. Use a file with two voices in it - a single " +
                 "speaker comes back nearly unchanged, because there is nothing to remove.")]
        public string InputWav = "Audio/bystander_off.wav";

        [Tooltip("Where the isolated file goes. Empty means Assets/Isolated/<name>_isolated.wav.")]
        public string OutputWav = "";

        [Header("Mode")]
        public Mode RunMode = Mode.File;

        [Tooltip("Streaming mode: samples handed to the pipeline at a time. 320 is 20 ms.")]
        [Range(64, 16000)] public int BlockSamples = 320;

        [Header("Status (read-only)")]
        public string Status = "idle";
        public float LoadSeconds;
        public double RealTimeFactor;

        [Tooltip("How much the output differs from the input. Near 1.0 means nothing was removed - either the " +
                 "input has only one voice, or the voice sample is not the caller.")]
        public float Similarity;

        [Range(0f, 1f)] public float Progress;

        private VoxLockSession _session;
        private Coroutine _job;
        private volatile bool _busy;
        private volatile string _error;
        private VoxLockFileResult _result;
        private string _outputPath;

        // 进度由工作线程写、主线程读，用普通字段加 volatile 就够：读到一个稍旧的值无所谓。
        private volatile float _progress;

        private void Start()
        {
            StartCoroutine(LoadModel());
        }

        private void OnDestroy()
        {
            StopRun();
            _session?.Dispose();
            _session = null;
        }

        private IEnumerator LoadModel()
        {
            Status = "loading " + VoxLockConfig.Describe(Precision);

            string enrollFile = VoxLockConfig.EnrollFileName(Precision);
            string chunkFile = VoxLockConfig.ChunkFileName(Precision);
            byte[] enroll = null, chunk = null;
            string error = null;

            VoxLockModelLoader.LoadAsync(enrollFile, (b, e) => { enroll = b; error = e; });
            while (enroll == null && error == null) yield return null;
            if (error != null) { Status = "failed: " + error; Debug.LogError("[VoxLock] " + error); yield break; }

            VoxLockModelLoader.LoadAsync(chunkFile, (b, e) => { chunk = b; error = e; });
            while (chunk == null && error == null) yield return null;
            if (error != null) { Status = "failed: " + error; Debug.LogError("[VoxLock] " + error); yield break; }

            float t0 = Time.realtimeSinceStartup;
            try
            {
                _session = VoxLockSession.Create(enroll, chunk, Precision, Threads);
            }
            catch (Exception e)
            {
                Status = "failed: " + e.Message;
                Debug.LogError("[VoxLock] " + e);
                yield break;
            }
            LoadSeconds = Time.realtimeSinceStartup - t0;

            Status = $"ready - {VoxLockConfig.Describe(Precision)}, loaded in {LoadSeconds:F1} s";
            Debug.Log($"[VoxLock] {Status}");

            StartRun();
        }

        /// <summary>处理当前配置的文件。可以从 UI 按钮调用。</summary>
        public void StartRun()
        {
            if (_session == null) { Status = "model not loaded yet"; return; }
            if (_busy) { Status = "already running"; return; }

            StopRun();

            string path = ResolveInput(InputWav);
            if (!File.Exists(path)) { Status = "no such file: " + path; return; }

            _outputPath = ResolveOutputPath(path);
            _result = default;
            _error = null;
            _progress = 0f;
            Progress = 0f;

            _job = StartCoroutine(RunOnWorker(path, _outputPath));
        }

        /// <summary>中止正在进行的处理。工作线程会在当前块结束时退出。</summary>
        public void StopRun()
        {
            if (_job != null) { StopCoroutine(_job); _job = null; }

            // 工作线程是后台线程，StopRun 之后它的结果会被丢弃；不 Join，避免卡住主线程。
            _busy = false;
        }

        private IEnumerator RunOnWorker(string inputPath, string outputPath)
        {
            bool streaming = RunMode == Mode.Streaming;
            int block = Math.Max(64, BlockSamples);
            var session = _session;

            Status = streaming
                ? $"streaming {Path.GetFileName(inputPath)} in {block}-sample blocks..."
                : $"isolating {Path.GetFileName(inputPath)}...";

            _busy = true;

            // 模型调用是纯计算，放到线程池上，主线程继续刷进度。
            var done = false;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                // Unity 会导入 Assets 下的 .wav。写到一半的文件头还是占位值，直接落成 .wav 会被导入成空
                // 音频并报错，所以先写成 .part——Unity 不认这个扩展名，看不到中间状态——写完再拷成正式文件。
                string temp = outputPath + ".part";
                try
                {
                    if (streaming)
                    {
                        _result = VoxLockAudio.IsolateFileStreaming(
                            session, inputPath, temp, block,
                            onProgress: (written, total) =>
                                _progress = total > 0 ? (float)Math.Min(1.0, written / total) : 0f);
                    }
                    else
                    {
                        _result = VoxLockAudio.IsolateFile(session, inputPath, temp);
                        _progress = 1f;
                    }

                    // 写完了再换成正式文件名。File.Move 的覆盖重载不在 Unity 的 .NET 版本里，先删再移。
                    if (File.Exists(outputPath)) File.Delete(outputPath);
                    File.Move(temp, outputPath);
                }
                catch (Exception e)
                {
                    _error = e.Message;
                    Debug.LogError("[VoxLock] " + e);
                }
                finally
                {
                    try { if (File.Exists(temp)) File.Delete(temp); } catch { /* 删不掉就留着，不影响结果 */ }
                }
                done = true;
            });

            while (!done)
            {
                Progress = _progress;
                Status = $"{RunMode} - {Progress * 100:F0}%";
                yield return null;
            }

            _busy = false;
            _job = null;

            if (_error != null)
            {
                Status = "failed: " + _error;
                yield break;
            }

            Progress = 1f;
            RealTimeFactor = _result.RealTimeFactor;
            Similarity = MeasureSimilarity(inputPath, outputPath);

            // 相关性接近 1 说明输出几乎等于输入——模型没找到要移除的东西。这几乎总是输入的问题，
            // 不是模型的问题：单说话人的录音本来就没有可移除的第二个人声。
            string note = Similarity > 0.98f
                ? "  (output ~= input: nothing to remove, use a file with two voices)"
                : "";

            Status = $"{_result} -> {outputPath}{note}";
            Debug.Log($"[VoxLock] {RunMode} {Status}");
        }

        /// <summary>
        /// 解析输入路径。绝对路径直接用；相对路径先当 <c>Assets/</c> 下的资源找——示例音频放在
        /// <c>Assets/Audio/</c>，在编辑器里点一下就能试听——找不到再按 StreamingAssets 的布局找
        /// （模型文件在那儿，旧位置也是那儿）。
        /// </summary>
        private static string ResolveInput(string path)
        {
            if (string.IsNullOrEmpty(path)) return path;
            if (Path.IsPathRooted(path)) return path;

            string underAssets = Path.Combine(Application.dataPath, path);
            return File.Exists(underAssets) ? underAssets : VoxLockModelLoader.Resolve(path);
        }

        /// <summary>
        /// 输出路径。留空就写到 <c>Assets/Isolated/&lt;原名&gt;_isolated.wav</c>：放在 Assets 下 Unity 才会把它
        /// 导成 AudioClip，在 Project 窗口里点一下就能试听。相对路径也按 Assets/ 解释，和 <see cref="InputWav"/>
        /// 保持一致。目录不存在就建出来。
        /// </summary>
        private string ResolveOutputPath(string inputPath)
        {
            string path;

            if (string.IsNullOrEmpty(OutputWav))
            {
                path = Path.Combine(Application.dataPath, OutputFolder,
                                    Path.GetFileNameWithoutExtension(inputPath) + "_isolated.wav");
            }
            else if (Path.IsPathRooted(OutputWav))
            {
                path = OutputWav;
            }
            else
            {
                path = Path.Combine(Application.dataPath, OutputWav);
            }

            string dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);

            return path;
        }

        /// <summary>默认输出目录，相对 <c>Assets/</c>。</summary>
        private const string OutputFolder = "Isolated";

        /// <summary>
        /// 输入和输出的相关系数，用来判断这次到底移除了多少东西。
        ///
        /// 隔离正常工作时会去掉一个人声，输出和输入的相关性明显低于 1（官方 demo 素材上约 0.76）。如果
        /// 拿到 0.99 以上，说明输出几乎就是输入的副本——输入里本来就只有一个人声，模型没有可移除的对象。
        /// 这个指标比用耳朵判断可靠：单说话人录音听起来"没效果"是对的，不是失败。
        /// </summary>
        private static float MeasureSimilarity(string inputPath, string outputPath)
        {
            try
            {
                float[] a = VoxLockWav.Read(inputPath, out int rateIn);
                float[] b = VoxLockWav.Read(outputPath, out _);

                // 输出总是 16 kHz。输入采样率不同就先重采样，否则两条曲线的时间轴对不上，相关系数没有意义。
                if (rateIn != VoxLockConfig.SampleRate)
                    a = VoxLockAudio.Resample(a, rateIn, VoxLockConfig.SampleRate);

                int n = Math.Min(a.Length, b.Length);
                if (n == 0) return 0f;

                double sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
                for (int i = 0; i < n; i++)
                {
                    double x = a[i], y = b[i];
                    sumA += x; sumB += y;
                    sumAA += x * x; sumBB += y * y;
                    sumAB += x * y;
                }

                double cov = n * sumAB - sumA * sumB;
                double va = n * sumAA - sumA * sumA;
                double vb = n * sumBB - sumB * sumB;
                if (va <= 0 || vb <= 0) return 0f;

                return (float)(cov / (Math.Sqrt(va) * Math.Sqrt(vb)));
            }
            catch (Exception e)
            {
                Debug.LogWarning("[VoxLock] could not compare input and output: " + e.Message);
                return 0f;
            }
        }
    }
}
