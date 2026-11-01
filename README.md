# gsv-tts-onnx

纯 C（C11）实现的 [GSV TTS](https://github.com/RVC-Boss/GPT-SoVITS) 本地推理引擎，编译为 Windows x86_64 DLL（`gsvtts.dll`）。脱离 Python / PyTorch，用 ONNX Runtime + 自研 C 侧 DSP/G2P 完成端到端语音合成，可经 P/Invoke 被 C# / C 调用。

- **模型**：v2ProPlus（32 kHz 输出；v4 / 48 kHz 预留）。仓库仅自带**底模**，其余音色按名加载、缺专属目录时回退底模。
- **语言**：中文（zh，含 RoBERTa BERT）、英文（en）、日文（ja，OpenJTalk）；支持跨语言合成。
- **接口**：[`include/gsvtts.h`](include/gsvtts.h)；模型包契约见 [`docs/MODEL_PACKAGE.md`](docs/MODEL_PACKAGE.md)。

## 目录结构

| 路径 | 说明 |
|---|---|
| `include/gsvtts.h` | 公开 C API（唯一对外头文件） |
| `src/` | 引擎主体：API / 模型加载 / GPT 生成 / SoVITS 解码 / BERT / G2P / ORT 会话封装；子目录 `dsp/`（DSP）、`g2p/`（zh/en/ja G2P） |
| `third_party/openjtalk/` | OpenJTalk 1.11 + MeCab 源码（ja G2P，CMake 直接编译） |
| `third_party/onnxruntime/` | ORT 头文件（DLL 需自行下载，见「构建」） |
| `tools/` | Python：ckpt/pth → ONNX 一键转换工具 |
| `project/` | C# P/Invoke 封装与示例（见 [`project/README.md`](project/README.md)） |
| `docs/` | MODEL_PACKAGE.md（模型包契约） |

## 合成流程

- **warmup**（每个 ref 一次，结果缓存）— `gsv_warmup_ref`：由 ref.wav + 文本 + 语言抽取并缓存 ge embedding、prompt codes、G2P phonemes（zh 另过 BERT）。
- **infer**（每次合成）— `gsv_infer`：目标文本 G2P → 与 prompt 拼接 → t2s GPT 自回归生成 codes（C 侧采样）→ SoVITS 解码为 s16le PCM。长文本自动分句；流式输出见 `gsv_stream_*`。

## 构建

依赖：mingw-w64 GCC（`-posix` 线程模型）、CMake ≥ 3.16、Ninja。

```bash
# 交叉编译（WSL/Linux）：工具链根目录默认 $HOME/opt/mingw，按本机安装位置调整
cmake -S . -B build -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Windows 原生构建：直接运行 `build-win.cmd`（mingw-w64 + Ninja；工具链 PATH 按其头部注释配置）。

产物：`build/gsvtts.dll`（并把 `onnxruntime*.dll` 拷到同目录）。

> **onnxruntime DLL（不入库）**：`third_party/onnxruntime/lib/*.dll`（~333 MB）已加入 `.gitignore`，构建前需自行下载放入该目录。ONNX Runtime **1.23.2** Windows x64：
> - CPU：[v1.23.2](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2) → `onnxruntime-win-x64-1.23.2.zip` → `lib/*.dll`
> - GPU：`onnxruntime-win-x64-gpu-1.23.2.zip`（多一个 `onnxruntime_providers_cuda.dll`；运行期另需 CUDA/cuDNN 运行时 DLL 与其同目录）

**ja G2P**：OpenJTalk 源码内置于 `third_party/openjtalk/`，CMake 直接编译（必需，缺失该目录时配置报错）。

DLL 运行期仅依赖 `KERNEL32` / `msvcrt`（w64devkit 原生构建实测；交叉编译构建可能引入 `libwinpthread-1.dll`，随 DLL 同目录分发即可）——自包含，无 libgcc/libstdc++ 运行时依赖。线程安全：单引擎的所有操作由内部锁串行化，可跨线程共享；需要并行吞吐时每线程各建一个引擎。

## 使用

C# 封装与端到端合成示例见 [`project/`](project/README.md)。

## 当前状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0–M3 | 引擎骨架 / ORT 封装 / manifest / ref warmup | ✅ |
| M4 | t2s GPT 自回归 + C 侧采样 | ✅ |
| M5 | SoVITS 解码链（codes→音频） | ✅ |
| M6 | 语言分发 G2P + `gsv_infer` 编排（zh/en/ja） | ✅ |
| M6b | zh RoBERTa BERT 逐音素特征 | ✅ 已对拍 |
| M7 | 流式合成 `gsv_stream_*` + 长文本分句 | ✅ 已对拍 |

底模 × zh/en/ja 端到端出有效 PCM（含跨语言 ja ref + en 目标）。

## 许可证

[MIT](LICENSE) © 2026 mbzmhs。第三方组件沿用其自身许可（`third_party/openjtalk/` — OpenJTalk/MeCab BSD 许可；`third_party/onnxruntime/` — ONNX Runtime MIT 许可），原始声明保留在各源码文件内。
