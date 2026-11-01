using GsvTts;

// ---------------------------------------------------------------------------
// Minimal end-to-end example for the gsvtts.dll C# wrapper.
//
// Build & run on Windows x64, with gsvtts.dll + onnxruntime*.dll next to the exe
// (or set OrtLibDir below). Adjust the paths at the top to your model package.
//
//   dotnet run --project GsvTts.Example
// ---------------------------------------------------------------------------

const string Manifest = @"C:\path\to\model-pkg\manifest.json";
const string RefWav   = @"C:\path\to\ref_neutral.wav";
const string OutDir   = @"C:\path\to\out";

// The reference transcript. IMPORTANT: this must be in the SAME language as the
// reference audio (the "ref" language), which is independent of the target language.
const string RefText  = "你好，今天天气不错，我们出去走走吧。"; // zh ref
const string RefLang  = "zh";

try
{
    using var engine = new GsvEngine(new GsvConfig
    {
        Device       = "auto",              // auto | cuda | cpu
        BaseManifest = Manifest,            // required: per-version base manifest.json
        // OrtLibDir  = OutDir,             // only if onnxruntime*.dll are not next to gsvtts.dll
    });

    Console.WriteLine($"gsvtts {GsvEngine.Version}");
    Console.WriteLine($"providers: {engine.AvailableProviders()}");
    Console.WriteLine($"state:     {engine.GetState()}");

    // 1) Load a character: (SoVITS voice dir, GPT model dir), resolved independently —
    //    either side falls back to that side's base model when its directory is absent.
    engine.LoadModel("my_voice", "s1v3");

    // 2) Warm up a reference (emotion/timbre preset). lang = the REF's own language.
    engine.WarmupRef("neutral", RefWav, RefText, RefLang);

    // 3) Synthesize + save WAVs.
    var zh = engine.Synthesize("你好，这是一条中文测试。", "zh", 1.0f, "neutral");
    zh.SaveWav($"{OutDir}\\csharp_zh.wav");
    Console.WriteLine($"zh: {zh.DurationSeconds:F2}s ({zh.FrameCount} samples) -> csharp_zh.wav");

    // Cross-language: English target with the zh ref — keep the ref language as "zh" (set at warmup).
    var en = engine.Synthesize("Hello world, this is an English test.", "en", 1.0f, "neutral");
    en.SaveWav($"{OutDir}\\csharp_en.wav");
    Console.WriteLine($"en: {en.DurationSeconds:F2}s ({en.FrameCount} samples) -> csharp_en.wav");

    // 4) Streaming synthesis (pull chunks as they are decoded).
    using (var stream = engine.BeginStream("流式合成测试。这是第二句。", "zh", 1.0f, "neutral"))
    {
        int i = 0;
        foreach (var chunk in stream.Chunks())
            Console.WriteLine($"stream chunk {i++}: {chunk.Length / 2} samples");
    }

    // 5) Optional: tune sampling parameters.
    engine.SetSampling(new GsvSampling { TopK = 8, NoiseScale = 0.3f });

    Console.WriteLine("OK");
    return 0;
}
catch (GsvException ex)
{
    Console.Error.WriteLine($"GSV error in {ex.Operation} (status={ex.StatusCode}): {ex.Message}");
    return 1;
}
catch (DllNotFoundException ex)
{
    Console.Error.WriteLine($"Could not load gsvtts.dll: {ex.Message}");
    Console.Error.WriteLine("Place gsvtts.dll next to the exe, or add its directory to the DLL search path.");
    return 2;
}
