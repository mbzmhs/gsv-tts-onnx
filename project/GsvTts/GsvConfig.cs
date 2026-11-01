using System.Globalization;
using System.Text;

namespace GsvTts
{
    /// <summary>
    /// Builder for the config JSON passed to <c>gsv_engine_create</c>. Set
    /// <see cref="BaseManifest"/> to load the base model set at init (required for real use).
    /// Unset optional fields are omitted from the JSON so the engine defaults apply.
    /// </summary>
    public sealed class GsvConfig
    {
        /// <summary>Device: "auto" (default), "cuda", or "cpu".</summary>
        public string Device { get; set; } = "auto";

        /// <summary>ORT intra-op threads. 0 = physical core count (default).</summary>
        public int IntraOpNumThreads { get; set; } = 0;

        /// <summary>Directory holding onnxruntime*.dll. Default: same dir as gsvtts.dll.</summary>
        public string? OrtLibDir { get; set; }

        /// <summary>
        /// Path to the per-version base manifest.json. Loaded once at init.
        /// Required for synthesis; omit only for a bare (model-less) engine.
        /// </summary>
        public string? BaseManifest { get; set; }

        /// <summary>fp16 model-variant preference: -1 auto (default), 0 off, 1 on.</summary>
        public int? UseFp16 { get; set; }

        /// <summary>Optional RNG seed for reproducible sampling (test hook).</summary>
        public int? RngSeed { get; set; }

        /// <summary>Optional inter-sentence silence in seconds (default 0.3 in the engine).</summary>
        public float? SentencePause { get; set; }

        /// <summary>Serialize to the config JSON string.</summary>
        public string ToJson()
        {
            var sb = new StringBuilder(256);
            bool first = true;

            void Str(string key, string val)
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append('"').Append(key).Append("\":\"").Append(Escape(val)).Append('"');
            }
            void Num(string key, string val)
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append('"').Append(key).Append("\":").Append(val);
            }

            sb.Append('{');
            Str("device", Device);
            Num("intra_op_num_threads", IntraOpNumThreads.ToString(CultureInfo.InvariantCulture));
            if (!string.IsNullOrEmpty(OrtLibDir)) Str("ort_lib_dir", OrtLibDir!);
            if (!string.IsNullOrEmpty(BaseManifest)) Str("base_manifest", BaseManifest!);
            if (UseFp16.HasValue) Num("use_fp16", UseFp16.Value.ToString(CultureInfo.InvariantCulture));
            if (RngSeed.HasValue) Num("rng_seed", RngSeed.Value.ToString(CultureInfo.InvariantCulture));
            if (SentencePause.HasValue) Num("sentence_pause", SentencePause.Value.ToString("0.###", CultureInfo.InvariantCulture));
            sb.Append('}');
            return sb.ToString();
        }

        // The C config parser unescapes backslash sequences (\\ -> \, \" -> ").
        // Escaping only these two is sufficient and safe for file paths.
        private static string Escape(string s)
        {
            if (s.IndexOfAny(new[] { '\\', '"' }) < 0) return s;
            var sb = new StringBuilder(s.Length + 8);
            foreach (char c in s)
            {
                if (c == '\\' || c == '"') sb.Append('\\');
                sb.Append(c);
            }
            return sb.ToString();
        }
    }
}
