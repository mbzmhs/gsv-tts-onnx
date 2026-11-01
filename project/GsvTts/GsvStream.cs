using System;
using System.Collections.Generic;
using System.IO;

namespace GsvTts
{
    /// <summary>
    /// A pull-based synthesis stream (sentence-granular chunks). Pull with <see cref="TryNext"/>
    /// until it reports done, or use the convenience <see cref="ReadAll"/> / <see cref="Chunks"/>.
    /// Call <see cref="Dispose"/> (or use in a <c>using</c> statement) to release native resources.
    /// </summary>
    public sealed class GsvStream : IDisposable
    {
        private readonly GsvEngine _engine;
        private IntPtr _handle;
        private readonly int _sampleRate;
        private bool _disposed;

        internal GsvStream(GsvEngine engine, IntPtr handle, int sampleRate)
        {
            _engine = engine;
            _handle = handle;
            _sampleRate = sampleRate;
        }

        /// <summary>Sample rate in Hz of the produced PCM.</summary>
        public int SampleRate => _sampleRate;

        /// <summary>
        /// Pull the next chunk into <paramref name="buffer"/> (a non-null short[]).
        /// Returns true with <paramref name="writtenSamples"/> &gt; 0 when a chunk was produced,
        /// false when the stream is finished. Throws <see cref="GsvException"/> on error.
        /// </summary>
        public bool TryNext(short[] buffer, out int writtenSamples)
        {
            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            int rc = NativeMethods.gsv_stream_next(_handle, buffer, buffer.Length, out long w);
            if (rc < 0) throw new GsvException("gsv_stream_next", rc, _engine.LastError);
            writtenSamples = (int)w;
            return rc != NativeMethods.StreamDone;   // true = got data, false = done
        }

        /// <summary>Read the entire remaining stream into one s16le PCM audio block.</summary>
        public GsvAudio ReadAll(int chunkSamples = 32000 * 8)
        {
            var buf = new short[Math.Max(1, chunkSamples)];
            using var ms = new MemoryStream();
            while (TryNext(buf, out int n))
            {
                if (n <= 0) continue;
                var chunk = new byte[n * 2];
                Buffer.BlockCopy(buf, 0, chunk, 0, n * 2);
                ms.Write(chunk, 0, chunk.Length);
            }
            return new GsvAudio(ms.ToArray(), _sampleRate);
        }

        /// <summary>Enumerate the stream as s16le PCM byte-array chunks until it is done.</summary>
        public IEnumerable<byte[]> Chunks(int chunkSamples = 32000 * 8)
        {
            var buf = new short[Math.Max(1, chunkSamples)];
            while (TryNext(buf, out int n))
            {
                if (n <= 0) continue;
                var chunk = new byte[n * 2];
                Buffer.BlockCopy(buf, 0, chunk, 0, n * 2);
                yield return chunk;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.gsv_stream_end(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~GsvStream() => Dispose();
    }
}
