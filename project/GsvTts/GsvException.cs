using System;

namespace GsvTts
{
    /// <summary>
    /// Raised when a native gsvtts.dll call returns an error status (a negative
    /// <c>gsv_status</c>). <see cref="StatusCode"/> carries the numeric code and the
    /// message includes the engine's own last-error text.
    /// </summary>
    public sealed class GsvException : Exception
    {
        /// <summary>The native status code (e.g. -3 = not loaded, -4 = ref missing).</summary>
        public int StatusCode { get; }

        /// <summary>Name of the operation that failed (the C function name).</summary>
        public string Operation { get; }

        internal GsvException(string operation, int status, string message)
            : base($"[{operation}] failed (status={status}): {message}")
        {
            Operation = operation;
            StatusCode = status;
        }
    }
}
