// VoxLock - 两个 ONNX 会话，以及它们之间的张量传递。
//
//   enroll.onnx   in:  enroll (1, T)                       out: enroll_n, k, v
//   chunk.onnx    in:  wav (1, S), enroll_n, k, v, h, c     out: frames (1, K, 512), h_out, c_out
//
// `enroll` 在 3 s 声音样本上只跑一次。`chunk` 在通话的每一帧上跑，并携带 LSTM 状态——这是唯一跨帧传递的
// 东西。形状从模型元数据里读，而不是写死，所以各个精度共用同一份代码。
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VoxLock
{
    /// <summary>
    /// <c>enroll</c> 从声音样本算出来的东西，整通电话期间一直保留。
    /// 这些张量直接引用 ONNX 的原生输出缓冲，所以只要这个 embedding 还活着，就必须持有拥有它的结果集合——
    /// 释放它就等于释放底层的 ONNX 值。
    /// </summary>
    public sealed class VoiceEmbedding : IDisposable
    {
        private readonly IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _owner;

        public readonly DenseTensor<float> EnrollN;
        public readonly DenseTensor<float> K;
        public readonly DenseTensor<float> V;

        internal VoiceEmbedding(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> owner,
                                DenseTensor<float> enrollN, DenseTensor<float> k, DenseTensor<float> v)
        {
            _owner = owner;
            EnrollN = enrollN;
            K = k;
            V = v;
        }

        public void Dispose()
        {
            _owner?.Dispose();
        }
    }

    /// <summary>加载好的一对 ONNX 会话。每个进程一个；流都从它创建，不要再建第二个。</summary>
    public sealed class VoxLockSession : IDisposable
    {
        public readonly InferenceSession Enroll;
        public readonly InferenceSession Chunk;
        public readonly VoxLockPrecision Precision;

        /// <summary>LSTM 状态的形状，从模型里读出来——随包发布的 v2 图是 (6, 32, 256)。</summary>
        public readonly int[] StateShape;
        public readonly int StateLength;

        VoxLockSession(InferenceSession enroll, InferenceSession chunk, VoxLockPrecision precision)
        {
            Enroll = enroll;
            Chunk = chunk;
            Precision = precision;

            StateShape = Chunk.InputMetadata["h"].Dimensions.ToArray();
            StateLength = 1;
            foreach (int d in StateShape) StateLength *= d;

            if (Chunk.InputMetadata.ContainsKey("context"))
            {
                throw new NotSupportedException(
                    "These are VoxLock 1.1 model files; the 1.2 code here needs the v2 graphs from " +
                    "https://huggingface.co/arbazsiddiqui/voxlock (onnx/v2).");
            }
        }

        /// <summary>从模型的原始字节加载两个图。<paramref name="threads"/>=0 时由 onnxruntime 自己决定。</summary>
        public static VoxLockSession Create(byte[] enrollModel, byte[] chunkModel,
                                            VoxLockPrecision precision = VoxLockPrecision.Int8, int threads = 0)
        {
            var enrollOptions = BuildOptions(threads);
            var chunkOptions = BuildOptions(threads);
            try
            {
                return new VoxLockSession(new InferenceSession(enrollModel, enrollOptions),
                                          new InferenceSession(chunkModel, chunkOptions), precision);
            }
            finally
            {
                enrollOptions.Dispose();
                chunkOptions.Dispose();
            }
        }

        private static SessionOptions BuildOptions(int threads)
        {
            var so = new SessionOptions();
            so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            if (threads > 0) so.IntraOpNumThreads = threads;
            if (threads != 1)
            {
                // 空闲线程去睡眠而不是自旋——参考实现也这么做，在实时通话里这是实打实的差别，
                // 因为 CPU 还要留给音频线程。
                so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
            }
            return so;
        }

        /// <summary>在声音样本上跑一次 <c>enroll</c>。样本至少要 3 s；用
        /// <see cref="VoxLockAudio.FitSample"/> 把它裁到正好 3 s。</summary>
        public VoiceEmbedding EnrollSample(float[] sample)
        {
            // 图要的是 (1, T)，不是 (T,)。
            var t = new DenseTensor<float>(new[] { 1, sample.Length });
            sample.AsSpan().CopyTo(t.Buffer.Span);

            var results = Enroll.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("enroll", t)
            });

            return new VoiceEmbedding(results,
                Of<DenseTensor<float>>(results, "enroll_n"),
                Of<DenseTensor<float>>(results, "k"),
                Of<DenseTensor<float>>(results, "v"));
        }

        /// <summary>
        /// 在 <paramref name="data"/> 中从 <paramref name="offset"/> 开始、长度为 <paramref name="length"/>
        /// 的样本上做一次模型调用，LSTM 状态通过 <paramref name="h"/> 和 <paramref name="c"/> 进出。
        /// 分离后的帧写进 <paramref name="frames"/>（行主序，共 <c>frameCount * 512</c> 个），并返回帧数。
        /// </summary>
        public int RunChunk(float[] data, int offset, int length, VoiceEmbedding voice,
                            float[] h, float[] c, float[] frames)
        {
            // 图要的是 (1, S)，不是 (S,)。
            var wav = new DenseTensor<float>(new[] { 1, length });
            data.AsSpan(offset, length).CopyTo(wav.Buffer.Span);

            var inputs = new List<NamedOnnxValue>(6)
            {
                NamedOnnxValue.CreateFromTensor("wav", wav),
                NamedOnnxValue.CreateFromTensor("enroll_n", voice.EnrollN),
                NamedOnnxValue.CreateFromTensor("k", voice.K),
                NamedOnnxValue.CreateFromTensor("v", voice.V),
                NamedOnnxValue.CreateFromTensor("h", new DenseTensor<float>(h, StateShape)),
                NamedOnnxValue.CreateFromTensor("c", new DenseTensor<float>(c, StateShape)),
            };

            using (var results = Chunk.Run(inputs))
            {
                int frameCount = 0;
                foreach (var output in results)
                {
                    switch (output.Name)
                    {
                        case "frames":
                            frameCount = CopyInto(output, frames);
                            break;
                        case "h_out":
                            CopyInto(output, h);
                            break;
                        case "c_out":
                            CopyInto(output, c);
                            break;
                    }
                }
                return frameCount;
            }
        }

        /// <summary>把输出张量拷进 <paramref name="dst"/>；快路径直接引用原生内存。</summary>
        private static int CopyInto(DisposableNamedOnnxValue value, float[] dst)
        {
            var tensor = value.AsTensor<float>();
            int n = (int)tensor.Length;

            var dense = tensor as DenseTensor<float>;
            if (dense != null)
            {
                dense.Buffer.Span.CopyTo(dst);
            }
            else
            {
                for (int i = 0; i < n; i++) dst[i] = tensor.GetValue(i);
            }
            return n / VoxLockConfig.FftSize;
        }

        private static T Of<T>(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, string name)
            where T : class
        {
            foreach (var v in results)
            {
                if (v.Name == name) return v.AsTensor<float>() as T;
            }
            throw new InvalidOperationException($"the enroll model has no output named {name}");
        }

        public void Dispose()
        {
            Enroll?.Dispose();
            Chunk?.Dispose();
        }
    }
}
