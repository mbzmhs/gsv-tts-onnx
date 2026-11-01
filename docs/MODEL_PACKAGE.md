# GSV TTS ONNX 模型包契约 (Model Package Contract) v1

C 推理引擎 (`gsvtts.dll`) 消费的 ONNX 模型包格式。导出由用户侧 Python 完成，两把一键工具：
**`tools/convert_sovits.py`**（GPT/t2s 共享 + SoVITS 逐音色，纯 fp16 I/O，可选 `--no-fp32`、
`--device auto|cuda|cpu` 优先 GPU、`--force` 重导）与 **`tools/convert_features.py`**
（共享基础：bert int8 / hubert / sv / ge / vq，全程 CPU，含 BERT 的 MatMul-only int8 量化），
本文件是双方的接口约定。

## dtype 策略（核心）

- **所有 `_fp16.onnx` 图都是「纯 fp16 I/O」**：float 输入**和**输出均为 fp16，整型张量保持
  int64（无 `keep_io_types` 的 fp32 边界包装）。C 端在边界做 f32↔f16 转换
  （`src/dsp/f16conv.h` + `gsv_in_float`/`gsv_out_f32`），t2s KV cache 用 `uint16_t*`。
- **混合 dtype（hybrid）**：
  - **SoVITS（enc_p / flowdec）双 dtype 并存**：`_fp32`（CPU）+ `_fp16`（CUDA）都随包分发。
    引擎按设备选变体（`src/engine.c sovits_variant()`：CPU→`_fp32`、CUDA→`_fp16`）。
    原因：ORT **CPU EP** 上 flowdec fp16 慢 ~1.7×、enc_p 慢 ~1.26×，故 CPU 走 fp32；CUDA 全支持 fp16。
  - **共享图（t2s_prefill / t2s_step / hubert / sv / ge）只发 `_fp16`**：CPU 与 CUDA EP 均能跑这些图的
    fp16（op 覆盖完整），故不另发 fp32。
- **BERT 保持 int8**（`bert_int8.onnx`，dynamic-MatMul 量化；I/O dtype 不变：int64 in / fp32 out）。

## 目录结构

```
<model_dir>/
  manifest.json                  # 全局元信息 + 各图参数（模型路径均为相对 pkg 根）
  gpt/                           # GPT 模型（按音色分子目录；缺失时回退到 manifest gpt.fallback 底模）
    <voice_name>/                # 可选：某音色的专属 GPT（用户自训音色按名加载）
      t2s_prefill_fp16.onnx
      t2s_step_fp16.onnx
    s1v3/                        # 当前底模（= manifest gpt.fallback；未配专属 GPT 的音色共用）
      t2s_prefill_fp16.onnx
      t2s_step_fp16.onnx
  features/                      # 特征 / 条件提取模型
    bert_int8.onnx               # zh BERT (hidden_states[-3])，int8；可选 fp32 回退 bert.onnx（当前瘦身包未随附）
    hubert_fp16.onnx             # wav16k -> last_hidden_state
    sv_fp16.onnx                 # fbank80 -> ERes2NetV2 embedding [20480]
    ge_fp16.onnx                 # mel704 + sv_emb -> ge[gin,1] + ge512[512,1]
  vq/                            # VQ 码本 + prompt-codes 投影（与 GPT 基座配套）
    codebook.bin                 # float32 [1024, 768] (行主序)
    ssl_proj.bin                 # Conv1d(768,768,k=2,s=2) 权重+偏置
  text/                          # BERT 分词器数据（C 端分词用）
    vocab.txt                    # BERT WordPiece 词表
    tokenizer.json               # （参考；C 端仅读 vocab.txt）
    token_snapshot.json          # （导出对拍快照）
  voices/
    <voice_name>/                # 逐音色 SoVITS；s2Gv2ProPlus = 底模（随包），其余为用户自训音色
      enc_p_fp32.onnx            # 文本+语义编码器+proj（合并，静态长度；CPU 用）
      enc_p_fp16.onnx            #   同上，纯 fp16（CUDA 用）
      flowdec_fp32.onnx          # flow 逆变换 + HiFi-GAN（动态长度；CPU 用）
      flowdec_fp16.onnx          #   同上，纯 fp16（CUDA 用）
  g2p/                           # C 侧 G2P 数据（按语言；dispatcher 按需加载，缺失则该语言不可用）
    zh/                          # jieba_dict.txt / pinyin.bin / hmm.bin / finalseg.bin / phrases.bin ...
    en/                          # cmu.bin namedict.bin pos_tag.bin wseg_uni.bin wseg_bi.bin gru.bin
                                   #   （6 个 .bin 已随本仓库 dist/g2p/en/ 分发，见下）
    ja/                          # OpenJTalk dicdir：sys.dic unk.dic matrix.bin rewrite.def char.bin *.def
```

> **路径约定**：manifest 里所有模型字段（`*.onnx`、`vq.codebook`、`bert.vocab` …）都是
> **相对 pkg 根的相对路径**。C 端一律 `path_join(pkg_dir, <relpath>)`。
> **基础特征图**（hubert/sv/ge/bert/vq）路径直接写在 manifest（指向 `_fp16.onnx`）。
> **GPT / SoVITS 逐音色图**由 `prefix` + 音色名派生，找不到则回退到 `fallback`（底模目录）：
>   - GPT：`<gpt.prefix>/<name>/t2s_{prefill,step}_fp16.onnx` → 回退 `<gpt.fallback>/…`（仅 fp16）。
>   - SoVITS：`<sovits.prefix>/<name>/{enc_p,flowdec}_<variant>.onnx` → 回退 `<sovits.fallback>/…`，
>     variant 由设备决定（CPU=`fp32`、CUDA=`fp16`），见 `src/engine.c resolve_named()`。
> 底模（fallback）= GPT `gpt/s1v3` + SoVITS `voices/s2Gv2ProPlus`；未配专属模型的音色自动落到底模。

> **g2p/en** 的 6 个 `.bin`（little-endian）已随本仓库 `dist/g2p/en/` 分发，打包模型包时复制到
> `<pkg>/g2p/en/` 即可。它们与 C 端 `#include` 的 `homographs_en.h` / `en_phonemes.h`
> （`src/g2p/`）是同一批源数据生成的配套产物，不可单独更换其一。

## manifest.json

```jsonc
{
  "contract": 1,
  "tts_version": "v2pp",          // 短 id 为准（"v2pp"）；旧长式 "v2ProPlus" 亦接受；v4 预留
  "gpt": {                        // GPT 逐音色目录 + 底模回退（纯 fp16）
    "prefix": "gpt",              //   <prefix>/<name>/t2s_{prefill,step}_fp16.onnx
    "fallback": "gpt/s1v3"        //   找不到专属 GPT 时回退到此底模目录
  },
  "sample_rate": 32000,           // 输出 wav 采样率 (v4=48000)
  "semantic_hz": 25,

  "t2s": {                        // GPT/t2s 结构参数（路径由 gpt.prefix/fallback 派生，不在此写死）
    "max_len": 1024,              // KV buffer 长度 (gpt_cache b==1 最大值)
    "num_layers": 24, "num_heads": 16, "model_dim": 512,
    "phoneme_vocab_size": 732,    // x_ids 取值范围（= GSV_N_SYMBOLS；旧值 367 是元数据 bug）
    "vocab_size": 1025,           // y_ids / logits 维度
    "eos": 1024,
    "suppressed": [280, 486, 1024]
  },

  "bert": {
    "onnx": "features/bert_int8.onnx",   // int8(dynamic-MatMul)；可选 fp32 回退 features/bert.onnx (fallback_onnx)
    "vocab": "text/vocab.txt",           // C 端分词词表（相对 pkg 根）
    "input_names": ["input_ids", "attention_mask"],   // int64
    "output_names": ["hidden_states"],                // float32 [B,L,1024] = hidden_states[-3]
    "dim": 1024,
    "max_length": 512,             // tokenizer truncation（元数据：C 端硬编码 512，见 src/bert_tok.c；改此值需重编）
    "use_for_langs": ["zh"],       // 仅 zh 段走 BERT，其它语言填零（元数据：C 端当前硬编码仅 zh；多语言预留）
    "dtype": "int8",
    "quant": "dynamic-int8-matmul",
    "fallback_onnx": "features/bert.onnx"   // C 端当前不使用（预留回退）
  },

  "hubert": {
    "onnx": "features/hubert_fp16.onnx",
    "input_names": ["wav16k"],     // fp16 [1, S] 原始波形 (已含 0.3s 尾部静音)
    "output_names": ["last_hidden_state"],  // fp16 [1, T', 768]
    "dim": 768, "dtype": "fp16"
  },

  "sv": {
    "onnx": "features/sv_fp16.onnx",
    "input_names": ["fbank"],      // fp16 [1, 80, T] Kaldi fbank (dither=0, sr=16000)
    "output_names": ["embedding"], // fp16 [20480]
    "dim": 20480, "dtype": "fp16"
  },

  "ge": {
    "onnx": "features/ge_fp16.onnx",
    "input_names": ["mel704", "sv_emb"],  // mel704: fp16 [1,704,Lm] (线性幅度谱前704行)
                                          // sv_emb: fp16 [20480]
    "output_names": ["ge", "ge512"],      // ge: fp16 [1, gin, 1]（flowdec 用）
                                          // ge512: fp16 [1, 512, 1]（= ge_to512(ge)，enc_p 用）
    "gin_channels": 1024, "dtype": "fp16"
  },

  "vq": {
    "codebook": "vq/codebook.bin",   // float32 [bins, dim]
    "dim": 768, "bins": 1024, "n_q": 1,
    "ssl_proj": {"kernel": 2, "stride": 2, "file": "vq/ssl_proj.bin"}  // Conv1d(768,768,k=2,s=2)
  },

  "sovits": {                     // SoVITS 逐音色目录 + 底模回退 + 版本级参数（来自 hps）
    "prefix": "voices",           //   <prefix>/<name>/{enc_p,flowdec}_<variant>.onnx
    "fallback": "voices/s2Gv2ProPlus",  // 找不到专属 SoVITS 时回退到此底模目录
    // v2pp 各音色共享同一套 hps（数值相同），故为版本级：
    "inter_channels": 192, "hidden_channels": 192, "filter_channels": 768,
    "n_heads": 2, "n_layers": 6, "kernel_size": 3,
    "gin_channels": 1024,
    "enc_p": {"ly_max": 1024, "lt_max": 512},   // 静态导出长度（见下）
    "samples_per_frame": 640,                    // prod(upsample_rates)
    "spec": {                                     // _get_spec 的 STFT 参数
      "filter_length": 2048, "hop_length": 640, "win_length": 2048
    }
  }
}
```

> `ly_max`/`lt_max` 由导出脚本按"最大句长"选定（默认 ly_max=1024 → 支持
> ≤512 个语义帧 ≈ 20.5s；超长句 C 端会报错提示重新导出更大 bucket）。

## 图 I/O 约定

> 下表列出**纯 fp16 变体**（`_fp16.onnx`）的 I/O dtype。SoVITS 的 `_fp32` 变体同形状、
> float 张量为 fp32；共享图无 fp32 变体。整型张量恒为 int64。

### t2s_prefill_fp16.onnx / t2s_step_fp16.onnx
- prefill: `x_ids[i64,1,Lx] y_ids[i64,1,Ly] bert[f16,1,Lx,1024] attn_bias[f16,1,1,S,S]`
  → `logits[f16,1,vocab] present_k[f16,L,1,H,S,D/H] present_v[f16,...]`
- step: `token_id[i64,1,1] pos_index[i64,1] idx[i64,T] past_k[f16,L,1,H,max_len,D/H] past_v[f16,...]`
  → `logits[f16,1,vocab] k_new[f16,L,1,H,1,D/H] v_new[f16,...]`

采样（top_k/top_p/temperature/repetition_penalty）在 C 端实现，不进图；KV cache 外置
（prefill 吃前缀、step 增量推进）。

### bert_int8.onnx
- 输入: `input_ids[i64,B,L] attention_mask[i64,B,L]`（动态 L ≤ max_length）
- 输出: `hidden_states[f32,B,L,1024]` = HF `output_hidden_states=True` 的 `[-3]`（**I/O dtype 不变**，仅内部 MatMul int8）
- C 端后处理（必须一致）:
  1. `tok = tokenizer(text)`：WordPiece，padding=True，truncation=512（lowercase/标点以 `tokenizer.json` 为准；
     C 端按 vocab.txt + basic+wordpiece 规则实现；导出脚本 dump 分词器快照供对拍）
  2. `feat = hidden[mask==1][1:-1]`（去 [CLS]/[SEP]）
  3. `bert_phones = repeat_interleave(feat, word2ph["ph"])`

### hubert_fp16.onnx
- 输入: `wav16k[f16,1,S]` 原始 16kHz 波形（动态 S；C 端已拼接 0.3s 静音）
- 输出: `last_hidden_state[f16,1,T',768]`
- C 端后续: `ssl_proj`(Conv1d k=2 s=2) → VQ encode(码本最近邻) → prompt codes（**信号处理保持 fp32**）

### sv_fp16.onnx
- 输入: `fbank[f16,1,80,T]`（Kaldi fbank：sr=16000, num_mel_bins=80, dither=0，
  默认 frame_length=25ms/frame_shift=10ms/preemphasis=0.97）
- 输出: `embedding[f16,20480]`（ERes2NetV2.forward3）

### ge_fp16.onnx
- 输入: `mel704[f16,1,704,Lm]`（动态 Lm；= 32k 波形 STFT 幅度谱前 704 行，STFT 参数见 sovits.spec，
  hann 窗，center=True reflect padding，power=1）；`sv_emb[f16,20480]`
- 输出: `ge[f16,1,gin,1]`（flowdec 用）和 `ge512[f16,1,512,1]`（= ge_to512(ge)，enc_p 用）
- 内部 = `ref_enc(mel704) + prelu(sv_emb_linear(sv_emb))`（v2ProPlus 路径）。
  **注**：ge 早期因图内结构性 Cast（int/bool mask）无法用 onnxconverter_common 转 fp16；改走
  **原生 torch fp16 导出**后由 torch 自动处理类型，故 ge 现已有 `_fp16` 变体。

### voices/<name>/enc_p_{fp32,fp16}.onnx  （静态长度，核心图：编码器+proj 合并）
输入（全部固定形状，C 端 pad 到满长；float dtype 随变体 fp32/fp16）:
- `quantized[f?,1,768,ly_max]`   # 码本查表 + 2x nearest 上采样后的语义特征
- `text_ids[i64,1,lt_max]`        # 目标文本 phoneme ids（pad=0）
- `ge512[f?,1,512,1]`            # ge_to512(ge)
- `y_mask[f?,1,1,ly_max]`        # 有效帧 mask（1/0）
- `t_mask[f?,1,1,lt_max]`
- `valid_len[i64]`               # = 2*T（T=语义帧数），驱动输出 Slice
输出:
- `m_p[f?,1,inter,L0]  logs_p[f?,1,inter,L0]  y_mask_out[f?,1,1,L0]`   # L0 = valid_len

内部 = ssl_proj → encoder_ssl(窗口注意力) → text_embedding → encoder_text(窗口)
→ MRTE(交叉注意力, 无窗口) → encoder2(窗口) → **narrow(valid_len)** → proj(1×1 conv, hidden→2*inter)
→ split(m_p, logs_p)。即旧 `encp_a`(编码器)+`encp_b`(proj) **合并为单图**，减少 host↔device 往返。
窗口相对位置表在固定长度下与真实长度运行逐位等价（偏移量只依赖 |i-j|），masked key 经 -1e4 掩码
softmax 后严格为 0。**速度插值 + 噪声留在 C 侧**：proj 是 1×1 conv（逐位置线性、不跨帧混叠）→
线性插值可与其交换（`interp(proj(y)) ≡ proj(interp(y))`），故把插值移到 enc_p 之后（作用于 m/logs/y_mask）
数值等价 torch，且免去图内动态 `Resize`。

### voices/<name>/flowdec_{fp32,fp16}.onnx  （动态长度）
- 输入: `z_p[f?,1,inter,L1] y_mask[f?,1,1,L1] ge[f?,1,gin,1]`（动态 L1）
- 输出: `audio[f?,1,1,·]`（长度由图输出动态给出，≈ L1 × 声码器上采样倍数；hps `samples_per_frame=640`）

## C 端 decode 流程（v2ProPlus，单句，2 图）

```
codes[T], phones[Lp], ge[gin] + ge512[512] (来自 ref 缓存)
1. quantized = codebook[codes]                      # [768,T]   (fp32, VQ 选码不受 fp16 影响)
2. quantized2 = repeat_each(quantized, 2)           # [768,2T]  (nearest x2)
3. enc_p(quantized2[·,·,ly_max], phones_pad, ge512, y_mask, t_mask, valid=2T)
    → m_p, logs_p, y_mask_out  @ L0=2T              # 变体按设备：CPU=_fp32 / CUDA=_fp16
4. if speed != 1: (m_p,logs_p,y_mask) = linear_interp(·, int(L0/speed)+1)   # C 侧，align_corners=False
5. z_p = m_p + randn * exp(logs_p) * noise_scale    # C 侧 RNG（可播种）
6. flowdec(z_p, y_mask, ge) → wav                   # 变体同 enc_p
```

## C 端 ref warmup 流程（gsv_warmup_ref）

```
wav (任意采样率 WAV)
A. ge 路径:
   a1. resample → 32k (Kaiser-window sinc FIR，对齐 torchaudio.Resample)     # fp32
   a2. peak norm: if max>1: audio /= min(2, max)
   a3. STFT(hann, spec 参数, center=reflect, power=1) → mel704 = spec[:, :704]  # fp32
   a4. resample(32k→16k) → Kaldi fbank80(dither=0) → sv_fp16.onnx → sv_emb[20480]
   a5. ge_fp16.onnx(mel704, sv_emb) → ge [gin,1] + ge512 [512,1]，两者均缓存
B. prompt codes 路径:
   b1. resample → 16k，尾部拼 0.3s 静音                                          # fp32
   b2. hubert_fp16.onnx(wav16k) → ssl [T',768]
   b3. ssl_proj(conv k=2 s=2) → VQ encode(码本最近邻) → prompt_codes[T'']        # fp32 选码
C. phones1/bert1: g2p(ref_text, ref_lang)（同推理路径）；zh 段过 bert_int8.onnx
```

## 数值对拍要求（导出脚本自检，见 convert_sovits.py）
- 每张图：ORT vs torch 同输入。fp32 变体 max-abs < 5e-3；fp16 变体相对误差 ~5e-3（容差放宽至 ~5e-2）。
- enc_p：至少 3 种 (ly, lt) 组合 pad 到满长运行，与 torch 真实长度对比，有效区达标。
- **纯 fp16 I/O 结构校验**：`_fp16` 图所有 float 输入/输出 dtype 必须为 fp16（int 保持 int64）。
- 整包端到端：同文本+同 ref+固定 RNG 种子，wav 与 torch 引擎 STFT 距离 < 0.01。

## 导出工具 `tools/convert_sovits.py`

一键从原型 torch 权重导出本契约格式（纯 fp16 I/O）：

```bash
# Windows Python（本机路径按实际填写），auto 设备优先 GPU、不足回退 CPU；GPU 上 trace ~2s/图
python tools/convert_sovits.py \
    --sovits SoVITS_weights_v2ProPlus/<voice>.pth [--sovits ...]   # 逐音色：enc_p_{fp32,fp16}+flowdec_{fp32,fp16}
    [--gpt models/s1v3.ckpt]                                       # 共享：t2s_prefill_fp16 + t2s_step_fp16
    --out <staging_dir>                                            # → <out>/voices/<name>/ 与 <out>/gpt/<stem>/
    [--no-fp32]                                                    # 仅 GPU 部署时跳过 fp32 版 enc_p/flowdec
    [--device auto|cuda|cpu] [--force] [--ly-max N --lt-max N]
```

- **SoVITS**：`enc_p_{fp32,fp16}.onnx` + `flowdec_{fp32,fp16}.onnx`（默认双 dtype；`--no-fp32` 只出 fp16）。
- **GPT/t2s**：`t2s_prefill_fp16.onnx` + `t2s_step_fp16.onnx`（仅 fp16）。
- 每张图 ORT↔torch 对拍 + `_fp16` 纯 fp16 I/O 结构校验；`--force` 删除目标后重导+重校验。

## 导出工具 `tools/convert_features.py`（共享基础）

一键导出契约的其余基础产物（全程 CPU）：

```bash
python tools/convert_features.py \
    --models-dir models/ \                       # HF 底模：chinese-roberta-wwm-ext-large、chinese-hubert-base、sv/
    [--sovits SoVITS_weights_v2ProPlus/<voice>.pth]   # ge 与 vq 从其 vq 提取；缺省则跳过这两步
    --out <pkg_root> \                           # → <out>/features/、<out>/text/、<out>/vq/
    [--keep-fp32] [--no-verify] [--force]
```

- **BERT**：fp32 导出并对拍 → `quantize_dynamic(weight_type=QInt8, op_types_to_quantize=["MatMul"])`
  → `features/bert_int8.onnx`（1198MB→367MB，I/O dtype 不变；特征 rel-L2 ~0.2 为该配方设计内水平，
  BERT 输出仅作条件特征，参考实现全链路实测有效）；`--keep-fp32` 保留 fp32 版作 `fallback_onnx`。
  同时产 `text/vocab.txt`（C 端分词词表）与 `token_snapshot.json` 等对拍参考。
- **hubert / sv / ge**：纯 fp16 I/O 导出 + 结构校验 + CPU-EP 数值校验（运行失败仅 warn，
  非有限值/超致命阈才 assert——CPU fp16 kernel 精度不能作为 CUDA 部署判据，见脚本 docstring）。
- **vq**：`codebook.bin`（fp32 [1024,768]）+ `ssl_proj.bin`（Conv1d k=2 s=2 权重+偏置），
  需 `--sovits`。
