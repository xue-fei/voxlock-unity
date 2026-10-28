// VoxLock - 从 StreamingAssets 里取出 ONNX 字节。
//
// 在编辑器和 Standalone 构建里 StreamingAssets 就是个普通文件夹，File.ReadAllBytes 就够了。在 Android 上
// 它在 APK 内部（jar: URL），在 WebGL 上是个 HTTP URL，两者都得走 UnityWebRequest。LoadAsync 驱动请求；
// LoadBytes 是编辑器/Standalone 下的阻塞快捷方式。
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace VoxLock
{
    public static class VoxLockModelLoader
    {
        /// <summary>模型文件的路径或 URL：StreamingAssets，加上 <see cref="VoxLockConfig.ModelFolder"/>。</summary>
        public static string PathFor(string fileName)
        {
            string root = Application.streamingAssetsPath.TrimEnd('/');
            string folder = VoxLockConfig.ModelFolder.Trim('/');
            return string.IsNullOrEmpty(folder) ? root + "/" + fileName : root + "/" + folder + "/" + fileName;
        }

        /// <summary>
        /// 文件实际所在的位置。先查配置的模型文件夹，再查这个工程用过的其它布局
        /// （StreamingAssets/voxlock、StreamingAssets 本身），这样挪动 ONNX 文件不需要改代码。在 Android
        /// 和 WebGL 上只查配置的文件夹——那边没有目录列举能力，请求一个不存在的文件只会失败。
        /// </summary>
        public static string Resolve(string fileName)
        {
            string preferred = PathFor(fileName);
            if (!CanReadDirectly) return preferred;

            if (File.Exists(preferred)) return preferred;

            string root = Application.streamingAssetsPath.TrimEnd('/');
            foreach (string folder in FallbackFolders)
            {
                string candidate = folder.Length == 0 ? root + "/" + fileName : root + "/" + folder + "/" + fileName;
                if (File.Exists(candidate)) return candidate;
            }
            return preferred;
        }

        /// <summary>在 <see cref="VoxLockConfig.ModelFolder"/> 之后尝试的布局，越具体越靠前。</summary>
        private static readonly string[] FallbackFolders = { "voxlock/v2", "voxlock", "" };

        /// <summary><see cref="Resolve"/> 会查的每一个位置，用于错误信息。</summary>
        public static string SearchedFor(string fileName)
        {
            string root = Application.streamingAssetsPath.TrimEnd('/');
            var parts = new List<string>();
            string folder = VoxLockConfig.ModelFolder.Trim('/');
            parts.Add(folder.Length == 0 ? root : root + "/" + folder);
            foreach (string f in FallbackFolders)
                parts.Add(f.Length == 0 ? root : root + "/" + f);
            return $"{fileName} in " + string.Join(", ", parts);
        }

        /// <summary>
        /// StreamingAssets 能否直接从磁盘读取。Android（APK 内）和 WebGL（HTTP）上为 false。
        /// </summary>
        public static bool CanReadDirectly =>
            Application.platform != RuntimePlatform.Android &&
            Application.platform != RuntimePlatform.WebGLPlayer;

        /// <summary>
        /// 阻塞加载。在编辑器和 Standalone 上可用；其它平台请用 <see cref="LoadAsync"/>。
        /// </summary>
        public static byte[] LoadBytes(string fileName)
        {
            string path = Resolve(fileName);
            if (!CanReadDirectly)
                throw new InvalidOperationException(
                    $"{Application.platform} keeps StreamingAssets off disk; use LoadAsync instead");

            if (!File.Exists(path))
                throw new FileNotFoundException("cannot find " + SearchedFor(fileName), path);

            return File.ReadAllBytes(path);
        }

        /// <summary>从 StreamingAssets 加载一个文件。<paramref name="onDone"/> 收到字节，或者错误信息。</summary>
        public static void LoadAsync(string fileName, Action<byte[], string> onDone)
        {
            if (onDone == null) throw new ArgumentNullException(nameof(onDone));

            string path = Resolve(fileName);
            if (CanReadDirectly)
            {
                try
                {
                    if (!File.Exists(path))
                    {
                        onDone(null, "cannot find " + SearchedFor(fileName));
                        return;
                    }
                    onDone(File.ReadAllBytes(path), null);
                }
                catch (Exception e)
                {
                    onDone(null, e.Message);
                }
                return;
            }

            LoadAsyncRequest(path, onDone);
        }

        private static void LoadAsyncRequest(string url, Action<byte[], string> onDone)
        {
            // 协程需要一个宿主；藏起来的临时宿主比让调用方自己维护一个更省事。
            var host = new GameObject("VoxLockModelLoader", typeof(ModelLoaderRunner));
            host.hideFlags = HideFlags.HideAndDontSave;
            var runner = (ModelLoaderRunner)host.GetComponent(typeof(ModelLoaderRunner));
            runner.Run(url, onDone);
        }

        private sealed class ModelLoaderRunner : MonoBehaviour
        {
            public void Run(string url, Action<byte[], string> onDone)
            {
                StartCoroutine(Request(url, onDone));
            }

            private System.Collections.IEnumerator Request(string url, Action<byte[], string> onDone)
            {
                using (var request = UnityWebRequest.Get(url))
                {
                    yield return request.SendWebRequest();

                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        onDone(null, $"{url}: {request.error}");
                    }
                    else
                    {
                        onDone(request.downloadHandler.data, null);
                    }
                }
                Destroy(gameObject);
            }
        }
    }
}
