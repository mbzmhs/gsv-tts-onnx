using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace GsvTts
{
    /// <summary>gsvtts.dll 原生库定位:把本程序集的 P/Invoke("gsvtts.dll")重定向到指定引擎目录。
    /// 用法:在首次调用任何 GsvEngine / GsvEngine.Version 之前调用一次
    /// <see cref="Configure"/>(不调用则走默认 DLL 搜索路径)。
    /// .NET 对每个程序集的 P/Invoke 解析器只能注册一次、且首次解析结果被缓存,
    /// 因此运行期更换 DLL 目录不会重新加载 —— 变更后需重启进程生效。</summary>
    public static class GsvNative
    {
        private static readonly object Lock = new();
        private static bool _registered;
        private static string? _dllDir; // 引擎 DLL 所在目录(空 = 走默认 DLL 搜索路径)

        public static string? DllDir
        {
            get { lock (Lock) return _dllDir; }
        }

        /// <summary>记录引擎目录并确保解析器已挂载。重复调用仅更新目录(已注册时)。</summary>
        public static void Configure(string? dllDir)
        {
            lock (Lock)
            {
                _dllDir = string.IsNullOrWhiteSpace(dllDir) ? null : dllDir.Trim();
                if (_registered)
                    return;
                try
                {
                    NativeLibrary.SetDllImportResolver(typeof(GsvNative).Assembly, Resolve);
                    _registered = true;
                }
                catch (InvalidOperationException)
                {
                    _registered = true; // 本程序集已注册过解析器(理论上只有这里注册)
                }
            }
        }

        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            // 仅拦截引擎本体;目录未配置或目录里没有该 DLL 时返回零指针 → 继续默认搜索路径
            if (!libraryName.Contains("gsvtts", StringComparison.OrdinalIgnoreCase))
                return IntPtr.Zero;
            var dir = DllDir;
            if (dir is null)
                return IntPtr.Zero;
            var candidate = Path.Combine(dir, libraryName);
            return File.Exists(candidate) ? NativeLibrary.Load(candidate, assembly, searchPath) : IntPtr.Zero;
        }
    }
}
