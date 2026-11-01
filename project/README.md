# GsvTts — C# wrapper for gsvtts.dll

A thin, safe **P/Invoke** wrapper around the GSV TTS ONNX engine (`gsvtts.dll`, a pure-C
Windows x64 DLL). It exposes the whole public C API with:

- UTF-8 string marshaling handled for you (no manual `IntPtr`/`byte[]` juggling),
- automatic output-buffer sizing for synthesis,
- `GsvException` on every error (with the engine's own last-error text),
- `IDisposable` lifecycle for the engine and streams,
- a built-in WAV writer so you can save results in one call.

> The C API contract is defined in [`../../include/gsvtts.h`](../../include/gsvtts.h).
> This wrapper mirrors it 1:1 (see [API reference](#api-reference)).

---

## Requirements

| What | Where |
|---|---|
| **Windows x64** | the DLL is a Windows binary (uses `GetFileAttributesA`, `LoadLibrary`, …) |
| **.NET 8 SDK** (or .NET 6/7; .NET Framework 4.8 with minor csproj tweaks) | to build |
| `gsvtts.dll` | the engine — from the C project's `build/` output |
| `onnxruntime.dll` (+ `onnxruntime_providers_*.dll`, CUDA runtime DLLs if using GPU) | next to `gsvtts.dll`, or point `GsvConfig.OrtLibDir` at them |

The wrapper library itself is platform-agnostic C#; it only needs the native DLLs **at runtime**.

---

## Project layout

```
project/
├─ GsvTts/                 # class library (the wrapper) — reference this from your app
│  ├─ GsvTts.csproj
│  ├─ NativeMethods.cs     #   raw P/Invoke bindings + gsv_status codes (internal)
│  ├─ GsvEngine.cs         #   managed engine lifecycle / inference (IDisposable)
│  ├─ GsvStream.cs         #   streaming synthesis (IDisposable)
│  ├─ GsvConfig.cs         #   config-JSON builder for engine creation
│  ├─ GsvAudio.cs          #   PCM container + WAV writer
│  ├─ GsvSampling.cs       #   sampling-parameter value object
│  └─ GsvException.cs      #   error type (carries the native status code)
├─ GsvTts.Example/         # runnable console example (end-to-end)
│  ├─ GsvTts.Example.csproj
│  └─ Program.cs
└─ README.md               # this file
```

---

## Build

From this `project/` directory, on Windows:

```powershell
# build just the wrapper library
dotnet build GsvTts\GsvTts.csproj

# build + run the example (also builds the library it references)
dotnet run --project GsvTts.Example
```

To use it in your own app, add a project reference to `GsvTts\GsvTts.csproj` (or copy the
built `GsvTts.dll`).

---

## Runtime deployment

The native DLLs must be locatable when the process runs. Easiest: put them next to your exe.

```
yourapp/
├─ yourapp.exe
├─ GsvTts.dll              # this wrapper (managed)
├─ gsvtts.dll              # the engine
└─ onnxruntime*.dll        # ONNX Runtime (+ CUDA provider DLLs for GPU)
```

If you keep `onnxruntime*.dll` somewhere else, set `GsvConfig.OrtLibDir` to that directory.
(`gsvtts.dll` is found via the normal Windows DLL search path; if it isn't next to your exe,
add its directory to the search path or copy it alongside.)

---

## Quick start

```csharp
using GsvTts;

using var engine = new GsvEngine(new GsvConfig
{
    Device       = "auto",                 // auto | cuda | cpu
    BaseManifest = @"C:\path\to\model-pkg\manifest.json",   // required for synthesis
});

// Load a character: (SoVITS voice dir, GPT model dir), resolved independently —
// either side falls back to that side's base model when its directory is absent.
engine.LoadModel("my_voice", "s1v3");

// Warm up a reference. lang = the REF's own language (NOT the target language).
engine.WarmupRef("neutral", @"C:\path\to\ref_neutral.wav",
                 "你好，今天天气不错，我们出去走走吧。", "zh");

GsvAudio audio = engine.Synthesize("你好，世界。", "zh", speed: 1.0f, refName: "neutral");
audio.SaveWav(@"out.wav");
Console.WriteLine($"{audio.DurationSeconds:F2}s @ {audio.SampleRate} Hz");
```

### Streaming

```csharp
using var stream = engine.BeginStream("第一句。第二句会更长一些。", "zh", 1.0f, "neutral");

// pull chunks as they decode (each is a byte[] of s16le mono PCM):
foreach (var chunk in stream.Chunks())
    SendToPlayer(chunk);                    // your audio sink

// ...or grab everything at once:
GsvAudio all = engine.BeginStream("…", "zh", 1.0f, "neutral").ReadAll();
```

---

## API reference

### `GsvEngine` (IDisposable) — one native engine instance

| Member | Description |
|---|---|
| `new GsvEngine(GsvConfig?)` / `new GsvEngine(string? configJson)` | Create. Pass a config with `BaseManifest` set. Throws on failure. |
| `static string Version` | Library version (e.g. `"0.2.0"`). |
| `string LastError` | Engine's last error text (`""` if none). |
| `int SampleRate` | Output sample rate (from state; 32000 for v2pp). Lazy/cached. |
| `void LoadModel(string sovitsName, string gptName)` | Load a character (single-resident). Each side resolves against the manifest independently (`voices/<sovitsName>`, `gpt/<gptName>`); a missing/empty side falls back to that side's base model. |
| `void UnloadModel()` | Release the loaded character (refs + sessions). Base persists. |
| `void WarmupRef(string refName, string wavPath, string? text, string? lang)` | Warm up one reference. **`lang` = the ref's own language.** |
| `void WarmupRefs(params (…)[])` | Warm up several references. |
| `void UnloadRef(string refName)` | Drop one cached reference. |
| `string GetState()` | JSON state snapshot (version, device, sample_rate, loaded_voice, refs, …). |
| `string AvailableProviders()` | Comma-joined ORT execution providers (no model load / VRAM needed). |
| `GsvAudio Synthesize(string text, string lang, float speed = 1.0f, string? refName = null)` | Full synthesis → PCM. Requires a warmed-up ref. Buffer auto-grows. |
| `GsvStream BeginStream(string text, string lang, float speed = 1.0f, string? refName = null)` | Begin streaming synthesis. Requires a warmed-up ref. |
| `void SetSampling(GsvSampling s)` / `GsvSampling GetSampling()` | Get/set GPT sampling + flow-noise params. |
| `void SetSentencePause(float seconds)` / `float GetSentencePause()` | Inter-sentence silence for multi-sentence text. |
| `void SetCutMute(float seconds)` / `float GetCutMute()` | Chunk-boundary silence base (seconds; ÷ speed, scaled by the chunk's tail punctuation). |
| `void SetCutMinlen(float chars)` / `float GetCutMinlen()` | Clause merge threshold (semantic chars; CJK=1, ASCII run=1.75). 0 = cut at every punctuation; a huge value disables cutting. |
| `void SetFp16(int mode)` | fp16 variant preference: -1 auto, 0 off (fp32), 1 on. Applies at `LoadModel`. |
| `long[] TextToPhonemeIds(string text, string lang)` | Text → phoneme id sequence (G2P only, no synthesis). |
| `void Dispose()` | Release the native engine. |

> **DLL 定位**:`gsvtts.dll` 默认走 .NET 标准 DLL 搜索路径(放 exe 旁即可)。若 DLL 在其他目录,
> 创建引擎前调用一次 `GsvNative.Configure(@"<引擎目录>")`(public static;解析器每程序集
> 只能注册一次,变更目录需重启进程)。

### `GsvStream` (IDisposable) — one streaming session

| Member | Description |
|---|---|
| `int SampleRate` | PCM sample rate. |
| `bool TryNext(short[] buffer, out int writtenSamples)` | Pull one chunk. `true` = got data, `false` = done. Throws on error. |
| `GsvAudio ReadAll(int chunkSamples = 32000*8)` | Drain the whole stream into one audio block. |
| `IEnumerable<byte[]> Chunks(int chunkSamples = 32000*8)` | Enumerate s16le PCM chunks until done. |
| `void Dispose()` | Release the native stream (safe after done). |

### `GsvConfig` — engine-creation config (→ JSON)

| Property | Default | Meaning |
|---|---|---|
| `Device` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"`. |
| `IntraOpNumThreads` | `0` | ORT intra-op threads (0 = physical cores). |
| `OrtLibDir` | — | Dir holding `onnxruntime*.dll` (default: next to `gsvtts.dll`). |
| `BaseManifest` | — | **Path to the per-version `manifest.json`.** Loaded once at init. Required for synthesis. |
| `UseFp16` | — | -1 auto / 0 off / 1 on (optional). |
| `RngSeed` | — | Reproducible-sampling seed (test hook). |
| `SentencePause` | — | Inter-sentence silence in seconds (optional). |

Unset optional properties are omitted from the JSON so engine defaults apply. Windows paths are
escaped automatically for the config parser.

### `GsvAudio` — PCM output

`Pcm` (byte[] s16le mono), `SampleRate`, `Channels` (=1), `BitsPerSample` (=16), `FrameCount`,
`DurationSeconds`, and `SaveWav(path)`.

### `GsvSampling` — sampling parameters

`TopK` (15), `TopP` (1.0), `Temperature` (1.0), `RepetitionPenalty` (1.35), `NoiseScale` (0.5).
Defaults match the reference server.

### `GsvException`

Thrown on any native error. `StatusCode` is the negative `gsv_status` code; the message includes
the operation name and the engine's last-error text. Common codes: `-2` invalid arg, `-3` not
loaded (call `LoadModel` first), `-4` ref missing (call `WarmupRef` first), `-6` file/IO,
`-7` ONNX Runtime backend error.

---

## Threading model

All operations on one engine are serialized by an internal lock — sharing a single `GsvEngine`
across threads is safe (calls queue; no data race). For parallel throughput, create **one engine
per thread** instead of sharing one. A `GsvStream` belongs to the engine that created it and
should be consumed on one thread at a time.

---

## ⚠️ Cross-language: set the *reference* language correctly

This is the #1 gotcha and it is **not** a bug in the wrapper or engine.

A reference (prompt) must be G2P'd in **its own language**, which is independent of the target
language you're synthesizing. Pass that language to `WarmupRef(..., lang)`:

- Generating **English** with a **Chinese** ref → `WarmupRef("neutral", zhWav, zhText, "zh")`,
  then `Synthesize(englishText, "en", …)`. Correct.
- If you instead warm the Chinese ref with `lang: "en"`, the prompt collapses to a few phonemes
  and the output is near-silence / very short. (Japanese refs are more forgiving because MeCab
  still segments CJK, which is why this usually shows up first for English.)

Best quality: use a reference whose audio + transcript are in the **same language as the target**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `DllNotFoundException` on first call | `gsvtts.dll` not found — put it next to your exe or add its dir to the DLL search path. |
| `GsvException … status=-7` (backend) at create/load | ONNX Runtime missing/mismatched — ensure `onnxruntime*.dll` are present (or set `OrtLibDir`). For CUDA, also stage the CUDA/cuDNN runtime DLLs. |
| `status=-3` (not loaded) on `Synthesize`/`WarmupRef` | Call `LoadModel(sovitsName, gptName)` first. |
| `status=-4` (ref missing) on `Synthesize` | Warm up that ref name first (`WarmupRef`). |
| Near-silent / very short audio (esp. English) | Wrong reference language — see the cross-language note above. |
| GPU not used though CUDA is installed | Set `Device = "cuda"` (or `"auto"`) and confirm `AvailableProviders()` lists `CUDAExecutionProvider`. The engine auto-falls back to CPU if a session can't be created on GPU. |

---

## Notes

- Audio is **mono 16-bit little-endian PCM** at the model's sample rate (32000 Hz for v2pp).
- `Synthesize` runs internal sentence-chunking for long text; use `BeginStream` when you want to
  feed audio incrementally.
- The wrapper does not copy or manage the model files — it only points at them via `BaseManifest`.
