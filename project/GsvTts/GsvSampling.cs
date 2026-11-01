namespace GsvTts
{
    /// <summary>
    /// GPT token sampling + flow-noise parameters. Defaults match the reference server:
    /// top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35, noise_scale=0.5.
    /// Apply with <see cref="GsvEngine.SetSampling"/>; read back with <see cref="GsvEngine.GetSampling"/>.
    /// </summary>
    public sealed class GsvSampling
    {
        /// <summary>Keep top-N tokens (0 = disabled, 1..N). Default 15.</summary>
        public int TopK { get; set; } = 15;

        /// <summary>Nucleus sampling threshold [0,1]. Default 1.0.</summary>
        public float TopP { get; set; } = 1.0f;

        /// <summary>Sampling temperature (>0). Default 1.0.</summary>
        public float Temperature { get; set; } = 1.0f;

        /// <summary>Repetition penalty (>=1 discourages repeats). Default 1.35.</summary>
        public float RepetitionPenalty { get; set; } = 1.35f;

        /// <summary>Flow noise scale: z = m + randn * exp(logs) * noise_scale. Default 0.5.</summary>
        public float NoiseScale { get; set; } = 0.5f;

        internal NativeMethods.GsvSamplingNative ToNative() => new NativeMethods.GsvSamplingNative
        {
            TopK = TopK,
            TopP = TopP,
            Temperature = Temperature,
            RepetitionPenalty = RepetitionPenalty,
            NoiseScale = NoiseScale,
        };

        internal static GsvSampling FromNative(NativeMethods.GsvSamplingNative n) => new GsvSampling
        {
            TopK = n.TopK,
            TopP = n.TopP,
            Temperature = n.Temperature,
            RepetitionPenalty = n.RepetitionPenalty,
            NoiseScale = n.NoiseScale,
        };
    }
}
