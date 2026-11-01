using System;
using System.Text;

namespace GsvTts
{
    /// <summary>
    /// Managed wrapper around a single gsvtts engine instance. Owns the native handle —
    /// call <see cref="Dispose"/> (or use in a <c>using</c> statement) to release it.
    ///
    /// Typical flow: create with a <see cref="GsvConfig"/> that sets <c>BaseManifest</c>,
    /// then <see cref="LoadModel"/> (a SoVITS×GPT pair), <see cref="WarmupRef"/>(s),
    /// and <see cref="Synthesize"/> / <see cref="BeginStream"/>.
    ///
    /// Thread-safety: all operations are serialized by the engine's internal lock, so one
    /// instance is safe to share across threads (calls queue). For parallel throughput,
    /// create one engine per thread instead of sharing.
    /// </summary>
    public sealed class GsvEngine : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;
        private int? _sampleRate;

        /// <summary>Create an engine from a config object (recommended).</summary>
        public GsvEngine(GsvConfig? config = null) : this(config?.ToJson())
        {
        }

        /// <summary>Create an engine from a raw config JSON string, or <c>null</c> for defaults.</summary>
        public GsvEngine(string? configJson)
        {
            _handle = NativeMethods.gsv_engine_create(configJson);
            if (_handle == IntPtr.Zero)
                throw new GsvException("gsv_engine_create", -1, NativeMethods.GsvGlobalLastError());
        }

        /// <summary>Library version string (e.g. "0.2.0").</summary>
        public static string Version => NativeMethods.GsvVersion();

        /// <summary>Last error message for this engine ("" if none).</summary>
        public string LastError => _handle == IntPtr.Zero ? "" : NativeMethods.GsvLastError(_handle);

        /// <summary>Output sample rate in Hz (from state; 32000 for v2pp). Lazy and cached.</summary>
        public int SampleRate
        {
            get
            {
                if (_sampleRate.HasValue) return _sampleRate.Value;
                _sampleRate = ParseInt(GetState(), "sample_rate") ?? 32000;
                return _sampleRate.Value;
            }
        }

        // ---- model lifecycle (character) ----

        /// <summary>
        /// Load a character: SoVITS 与 GPT 按各自目录名独立解析(模型包 voices/&lt;sovitsName&gt;、
        /// gpt/&lt;gptName&gt;)。空名字或该侧目录不存在时回退到 manifest 的底模 —— 因此任意
        /// SoVITS×GPT 组合都无需在模型包内创建联接目录。The base set must already be loaded
        /// via <c>BaseManifest</c>. Loading replaces any previously loaded character (single-resident).
        /// </summary>
        public void LoadModel(string sovitsName, string gptName) =>
            Check(NativeMethods.gsv_load_model(_handle, sovitsName ?? "", gptName ?? ""), "gsv_load_model");

        /// <summary>Release the loaded character (refs + its sessions). The base set persists.</summary>
        public void UnloadModel() =>
            Check(NativeMethods.gsv_unload_model(_handle), "gsv_unload_model");

        // ---- reference (emotion preset) cache ----

        /// <summary>
        /// Warm up a reference: computes prompt semantic codes + emotion embedding from
        /// <paramref name="wavPath"/> (any-rate WAV) and its transcript. <paramref name="lang"/>
        /// is the REF's own language (independent of the synthesis target language).
        /// </summary>
        public void WarmupRef(string refName, string wavPath, string? text, string? lang) =>
            Check(NativeMethods.gsv_warmup_ref(_handle,
                Require(refName, "refName"), Require(wavPath, "wavPath"), text ?? "", lang ?? ""),
                "gsv_warmup_ref");

        /// <summary>Warm up several references (each acquires the engine lock in turn).</summary>
        public void WarmupRefs(params (string refName, string wavPath, string? text, string? lang)[] refs)
        {
            if (refs == null || refs.Length == 0) throw new ArgumentException("no refs given", nameof(refs));
            foreach (var r in refs) WarmupRef(r.refName, r.wavPath, r.text, r.lang);
        }

        /// <summary>Drop one cached reference.</summary>
        public void UnloadRef(string refName) =>
            Check(NativeMethods.gsv_unload_ref(_handle, Require(refName, "refName")), "gsv_unload_ref");

        // ---- state ----

        /// <summary>JSON state snapshot (version, device, sample_rate, loaded_voice, refs, ...).</summary>
        public string GetState()
        {
            var buf = new byte[4096];
            int rc = NativeMethods.gsv_get_state(_handle, buf, buf.Length, out int req);
            if (rc < 0) throw new GsvException("gsv_get_state", rc, LastError);
            if (req > buf.Length)
            {
                buf = new byte[req];
                NativeMethods.gsv_get_state(_handle, buf, buf.Length, out req);
            }
            return Encoding.UTF8.GetString(buf, 0, Math.Max(0, Math.Min(req, buf.Length)));
        }

        /// <summary>Comma-joined ORT execution providers available in this process.</summary>
        public string AvailableProviders()
        {
            var buf = new byte[512];
            int rc = NativeMethods.gsv_available_providers(_handle, buf, buf.Length);
            if (rc < 0) throw new GsvException("gsv_available_providers", rc, LastError);
            return Encoding.UTF8.GetString(buf).TrimEnd('\0');
        }

        // ---- inference ----

        /// <summary>
        /// Synthesize the full text to mono s16le PCM (internal sentence chunking for long text).
        /// Requires a previously warmed-up reference. The output buffer is grown automatically if needed.
        /// </summary>
        public GsvAudio Synthesize(string text, string lang, float speed = 1.0f, string? refName = null)
        {
            Require(text, "text");
            Require(lang, "lang");
            if (string.IsNullOrEmpty(refName))
                throw new ArgumentException("a warmed-up reference is required: call WarmupRef(...) first and pass its name", nameof(refName));

            long cap = 32000L * 60;   // ~1 min at 32 kHz; re-run once if the result is longer
            var buf = new short[(int)cap];
            int rc = NativeMethods.gsv_infer(_handle, text, lang, speed, refName, buf, buf.Length, out long written);
            Check(rc, "gsv_infer");

            if (written > buf.Length)
            {
                buf = new short[(int)Math.Min(written + 32000L, int.MaxValue)];
                rc = NativeMethods.gsv_infer(_handle, text, lang, speed, refName, buf, buf.Length, out written);
                Check(rc, "gsv_infer");
            }

            var pcm = new byte[(int)(written * 2)];
            Buffer.BlockCopy(buf, 0, pcm, 0, pcm.Length);
            return new GsvAudio(pcm, SampleRate);
        }

        /// <summary>Begin a pull-based streaming synthesis. Requires a warmed-up reference.</summary>
        public GsvStream BeginStream(string text, string lang, float speed = 1.0f, string? refName = null)
        {
            Require(text, "text");
            Require(lang, "lang");
            if (string.IsNullOrEmpty(refName))
                throw new ArgumentException("a warmed-up reference is required", nameof(refName));

            var h = NativeMethods.gsv_stream_begin(_handle, text, lang, speed, refName);
            if (h == IntPtr.Zero) throw new GsvException("gsv_stream_begin", -1, LastError);
            return new GsvStream(this, h, SampleRate);
        }

        // ---- sampling / parameters ----

        /// <summary>Set GPT token-sampling + flow-noise parameters.</summary>
        public void SetSampling(GsvSampling s)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));
            var n = s.ToNative();
            Check(NativeMethods.gsv_set_sampling(_handle, in n), "gsv_set_sampling");
        }

        /// <summary>Read the current sampling parameters.</summary>
        public GsvSampling GetSampling()
        {
            Check(NativeMethods.gsv_get_sampling(_handle, out var n), "gsv_get_sampling");
            return GsvSampling.FromNative(n);
        }

        /// <summary>Set inter-sentence silence (seconds) for multi-sentence synthesis.</summary>
        public void SetSentencePause(float seconds) =>
            Check(NativeMethods.gsv_set_sentence_pause(_handle, seconds), "gsv_set_sentence_pause");

        /// <summary>Get the current inter-sentence silence (seconds).</summary>
        public float GetSentencePause()
        {
            Check(NativeMethods.gsv_get_sentence_pause(_handle, out var s), "gsv_get_sentence_pause");
            return s;
        }

        /// <summary>Set the chunk-boundary silence base (seconds; ÷ speed, scaled by the
        /// chunk's tail punctuation — 分块标点停顿的本体,真实静音在每块合成后追加)。</summary>
        public void SetCutMute(float seconds) =>
            Check(NativeMethods.gsv_set_cut_mute(_handle, seconds), "gsv_set_cut_mute");

        /// <summary>Get the chunk-boundary silence base (seconds).</summary>
        public float GetCutMute()
        {
            Check(NativeMethods.gsv_get_cut_mute(_handle, out var s), "gsv_get_cut_mute");
            return s;
        }

        /// <summary>Set the clause merge threshold (semantic chars; CJK=1, ASCII word=1.75)。
        /// 0 = 在每个标点处都切块;调大 = 更少切块。</summary>
        public void SetCutMinlen(float chars) =>
            Check(NativeMethods.gsv_set_cut_minlen(_handle, chars), "gsv_set_cut_minlen");

        /// <summary>Get the clause merge threshold (semantic chars).</summary>
        public float GetCutMinlen()
        {
            Check(NativeMethods.gsv_get_cut_minlen(_handle, out var c), "gsv_get_cut_minlen");
            return c;
        }

        /// <summary>Set fp16 model-variant preference: -1 auto, 0 off (fp32), 1 on. Applies at LoadModel time.</summary>
        public void SetFp16(int mode) =>
            Check(NativeMethods.gsv_set_fp16(_handle, mode), "gsv_set_fp16");

        // ---- G2P helper ----

        /// <summary>Text -> final phoneme id sequence (same G2P path as synthesis).</summary>
        public long[] TextToPhonemeIds(string text, string lang)
        {
            Require(text, "text");
            Require(lang, "lang");
            int cap = Math.Max(64, text.Length * 8);
            var ids = new long[cap];
            int rc = NativeMethods.gsv_text_to_phoneme_ids(_handle, text, lang, ids, ids.Length, out int n);
            Check(rc, "gsv_text_to_phoneme_ids");
            if (n > ids.Length) n = ids.Length;   // defensive: never read past the buffer
            var result = new long[n];
            Buffer.BlockCopy(ids, 0, result, 0, n * 8);
            return result;
        }

        // ---- helpers ----

        private void Check(int rc, string op)
        {
            if (rc < 0) throw new GsvException(op, rc, LastError);
        }

        private static string Require(string s, string what) =>
            string.IsNullOrEmpty(s) ? throw new ArgumentException($"{what} must be non-empty.", what) : s;

        private static int? ParseInt(string json, string key)
        {
            if (string.IsNullOrEmpty(json)) return null;
            var pat = "\"" + key + "\":";
            int i = json.IndexOf(pat, StringComparison.Ordinal);
            if (i < 0) return null;
            i += pat.Length;
            while (i < json.Length && (json[i] == ' ' || json[i] == '\t')) i++;
            int start = i;
            while (i < json.Length && (char.IsDigit(json[i]) || json[i] == '-')) i++;
            if (i > start && int.TryParse(json.Substring(start, i - start), out var v)) return v;
            return null;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.gsv_engine_destroy(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~GsvEngine() => Dispose();
    }
}
