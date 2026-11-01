using System;
using System.Runtime.InteropServices;

namespace GsvTts
{
    /// <summary>
    /// Raw P/Invoke bindings to gsvtts.dll (C ABI). All strings are UTF-8.
    /// Internal use — prefer the managed wrappers <see cref="GsvEngine"/> and <see cref="GsvStream"/>,
    /// which add error handling, buffer management and disposal.
    /// </summary>
    internal static class NativeMethods
    {
        private const string Lib = "gsvtts.dll";

        // ---- status codes (mirror the C gsv_status enum) ----
        public const int Ok = 0;
        public const int ErrGeneric = -1;
        public const int ErrInvalidArg = -2;
        public const int ErrNotLoaded = -3;
        public const int ErrRefMissing = -4;
        public const int ErrNoMemory = -5;
        public const int ErrIo = -6;
        public const int ErrBackend = -7;
        public const int ErrTextEmpty = -8;
        public const int ErrBusy = -9;

        // Positive (non-error) return code from gsv_stream_next.
        public const int StreamDone = 1;

        /// <summary>Mirrors the C <c>gsv_sampling</c> struct (sequential, blittable).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct GsvSamplingNative
        {
            public int TopK;
            public float TopP;
            public float Temperature;
            public float RepetitionPenalty;
            public float NoiseScale;
        }

        // ---- lifecycle ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr gsv_engine_create(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? config_json);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void gsv_engine_destroy(IntPtr e);

        // 注意:gsv_last_error / gsv_global_last_error / gsv_version 返回的都是 DLL 内部
        // 静态缓冲区的指针。这些 P/Invoke 绝不能声明成 string 返回值 —— .NET 互操作会在
        // 拷贝之后用 CoTaskMemFree 释放返回指针,等于对 DLL 的静态内存做堆释放,
        // 直接堆损坏(0xc0000374,崩不崩取决于该地址相对堆段的 ASLR 布局)。
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr gsv_last_error(IntPtr e);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr gsv_global_last_error();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr gsv_version();

        /// <summary>读取引擎级错误消息(UTF-8 静态缓冲区,拷贝为托管字符串)。</summary>
        public static string GsvLastError(IntPtr e) => PtrToUtf8(gsv_last_error(e));

        /// <summary>读取全局错误消息(创建引擎失败时引擎句柄尚不存在,用它取原因)。</summary>
        public static string GsvGlobalLastError() => PtrToUtf8(gsv_global_last_error());

        /// <summary>库版本字符串(如 "0.2.0")。</summary>
        public static string GsvVersion() => PtrToUtf8(gsv_version());

        private static string PtrToUtf8(IntPtr p) => Marshal.PtrToStringUTF8(p) ?? "";

        // ---- model lifecycle (character) ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_load_model(IntPtr e,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string sovits_name,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string gpt_name);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_unload_model(IntPtr e);

        // ---- reference (emotion preset) cache ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_warmup_ref(IntPtr e,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string ref_name,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string wav_path,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? text,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? lang);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_unload_ref(IntPtr e,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string ref_name);

        // ---- state ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_get_state(IntPtr e, byte[] buf, int buf_size, out int required);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_available_providers(IntPtr e, byte[] buf, int buf_size);

        // ---- inference ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_infer(IntPtr e,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lang,
            float speed,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string ref_name,
            short[] out_pcm, long out_capacity, out long written_samples);

        // ---- streaming ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr gsv_stream_begin(IntPtr e,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lang,
            float speed,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string ref_name);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_stream_next(IntPtr s, short[] out_pcm, long out_capacity, out long written);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void gsv_stream_end(IntPtr s);

        // ---- sampling / parameters ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_set_sampling(IntPtr e, in GsvSamplingNative s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_get_sampling(IntPtr e, out GsvSamplingNative s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_set_sentence_pause(IntPtr e, float seconds);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_get_sentence_pause(IntPtr e, out float seconds);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_set_cut_mute(IntPtr e, float seconds);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_get_cut_mute(IntPtr e, out float seconds);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_set_cut_minlen(IntPtr e, float chars);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_get_cut_minlen(IntPtr e, out float chars);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_set_fp16(IntPtr e, int mode);

        // ---- G2P helper ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gsv_text_to_phoneme_ids(IntPtr e,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lang,
            long[] out_ids, int cap, out int out_n);
    }
}
