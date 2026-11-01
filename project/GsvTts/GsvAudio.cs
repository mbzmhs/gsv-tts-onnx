using System;
using System.IO;

namespace GsvTts
{
    /// <summary>
    /// Decoded mono PCM output (little-endian 16-bit) at a given sample rate.
    /// Use <see cref="SaveWav"/> to write a standard WAV file, or consume <see cref="Pcm"/> directly.
    /// </summary>
    public sealed class GsvAudio
    {
        /// <summary>Raw interleaved s16le PCM bytes (mono).</summary>
        public byte[] Pcm { get; }

        /// <summary>Output sample rate in Hz (32000 for v2pp/v2ProPlus, 48000 for v4).</summary>
        public int SampleRate { get; }

        public int Channels => 1;
        public int BitsPerSample => 16;

        /// <summary>Number of PCM samples (frames).</summary>
        public int FrameCount => Pcm.Length / 2;

        /// <summary>Duration in seconds.</summary>
        public double DurationSeconds => SampleRate > 0 ? (double)FrameCount / SampleRate : 0.0;

        public GsvAudio(byte[] pcm, int sampleRate)
        {
            Pcm = pcm ?? throw new ArgumentNullException(nameof(pcm));
            SampleRate = sampleRate;
        }

        /// <summary>Write to a standard 16-bit PCM WAV file (overwrites any existing file).</summary>
        public void SaveWav(string path)
        {
            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs, System.Text.Encoding.ASCII, leaveOpen: true);

            int byteRate = SampleRate * Channels * BitsPerSample / 8;
            short blockAlign = (short)(Channels * BitsPerSample / 8);
            int dataSize = Pcm.Length;

            // RIFF header
            bw.Write(Ascii("RIFF"));
            bw.Write(36 + dataSize);          // file size - 8
            bw.Write(Ascii("WAVE"));
            bw.Write(Ascii("fmt "));
            bw.Write(16);                     // fmt chunk size (PCM)
            bw.Write((short)1);               // audio format: 1 = PCM
            bw.Write((short)Channels);
            bw.Write(SampleRate);
            bw.Write(byteRate);
            bw.Write(blockAlign);
            bw.Write((short)BitsPerSample);
            bw.Write(Ascii("data"));
            bw.Write(dataSize);
            bw.Write(Pcm);
        }

        private static byte[] Ascii(string s)
        {
            var b = new byte[s.Length];
            for (int i = 0; i < s.Length; i++) b[i] = (byte)s[i];
            return b;
        }
    }
}
