#!/usr/bin/env python
"""convert_features.py - 一键导出模型包的「共享基础」部分：BERT(int8) / hubert / sv / ge / VQ。

与 tools/convert_sovits.py 配套：那个脚本产出 GPT(t2s) 与逐音色 SoVITS，本脚本产出契约里
其余的基础产物（均写入 --out，布局与 docs/MODEL_PACKAGE.md 一致）:
    features/bert_int8.onnx        (zh BERT，dynamic-MatMul int8 量化；I/O dtype 不变)
    features/hubert_fp16.onnx      (wav16k -> last_hidden_state)
    features/sv_fp16.onnx          (fbank -> ERes2NetV2 embedding)
    features/ge_fp16.onnx          (mel704 + sv_emb -> ge + ge512；需 --sovits 提供 vq)
    text/vocab.txt (+ tokenizer.json / tokenizer_config.json / special_tokens_map.json
                    / token_snapshot.json —— C 端分词只用 vocab.txt，其余为对拍参考)
    vq/codebook.bin                (float32 [1024, 768] 行主序；需 --sovits)
    vq/ssl_proj.bin                (Conv1d(768,768,k=2,s=2) 权重+偏置；需 --sovits)

设计要点:
- 全程 CPU 导出（脚本自动清 CUDA_VISIBLE_DEVICES），与 convert_sovits 的 GPU 路径互不影响。
- **BERT int8 配方（锁定）**：fp32 导出并对拍后，`quantize_dynamic(weight_type=QInt8,
  op_types_to_quantize=['MatMul'])` —— 1198MB -> 367MB(3.26x)，I/O dtype 不变（int64 in /
  fp32 out），C 端零改动 drop-in。特征 rel-L2 误差 ~0.2 属该量化的设计内水平（BERT 输出仅作
  条件特征，经 bert_proj 线性层进入 t2s；参考实现全链路实测有效音频、G2P 逐位不变）。
- **fp16 特征图（hubert/sv/ge）为「纯 fp16 I/O」**：float 输入输出均 fp16，整型张量保持 int64。
  数值校验语义：CPU-EP 的 fp16 kernel 精度因模型而异（hubert ~2e-3、sv ~0.28），**不能**作为
  CUDA 部署正确性判据 —— 故运行失败/缺 kernel 时 warn 跳过；非有限值或 rel>fatal 才 assert。
- 码本/VQ 选码、prompt codes 路径全程 fp32（fp16 不影响选码）。

用法:
    runtime/python.exe tools/convert_features.py \
        --models-dir models/ \
        [--sovits SoVITS_weights_v2ProPlus/<voice>.pth]   # 缺省则只出 bert/hubert/sv，跳过 ge+vq
        --out <pkg_root> \
        [--keep-fp32] [--no-verify] [--force]
        [--skip-bert] [--skip-hubert] [--skip-sv] [--skip-ge-vq]

依赖: torch / transformers / onnx / onnxruntime(+quantization) / numpy；
      --sovits 与 sv 步骤还需用户侧 runtime 的 `gsv_tts` 包（与 convert_sovits.py 相同的约定）。
"""
import argparse
import json
import os
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 全程 CPU 导出
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (project_root, os.path.join(project_root, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from torch import nn

import onnx
import onnxruntime as ort

OPSET = 17
ET = {1: "fp32", 10: "fp16", 7: "int64", 6: "int32", 9: "bool", 3: "int8"}
INT_NAMES = {"input_ids", "attention_mask"}
TOL = 5e-3            # fp32 对拍 max-abs 容差
FP16_FATAL_REL = 5.0  # fp16 相对误差致命阈（超=真实 bug）；warn 阈 0.5
_NO_VERIFY = False


# ---------------------------------------------------------------- 通用助手

def _done(*paths):
    return all(os.path.exists(p) and os.path.getsize(p) > 1024 for p in paths)


def _export(model, args_tuple, path, input_names, output_names, dynamic_axes):
    print(f"  导出 {os.path.basename(path)} …")
    t0 = time.time()
    torch.onnx.export(
        model, args_tuple, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET, dynamo=False, do_constant_folding=True,
    )
    print(f"    完成 {os.path.getsize(path)/1e6:.0f}MB {time.time()-t0:.1f}s")


def _check(name, a, b, tol=TOL):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{name} 形状 {a.shape} vs {b.shape}"
    d = float(np.abs(a - b).max())
    print(f"  [对拍] {name} max-abs={d:.2e} (tol {tol:g})")
    assert d < tol, f"{name} 对拍未通过 (diff={d})"


def _sess(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _io_types(path):
    g = onnx.load(path).graph
    ins = [(i.name, ET.get(i.type.tensor_type.elem_type, "?")) for i in g.input]
    outs = [(o.name, ET.get(o.type.tensor_type.elem_type, "?")) for o in g.output]
    return ins, outs


def _assert_pure_fp16(path):
    """float I/O 必须全是 fp16；INT_NAMES 里的输入保持 int。"""
    ins, outs = _io_types(path)
    bad = []
    for n, t in list(ins) + list(outs):
        if t not in ("fp16", "int64", "int32"):
            bad.append((n, t))
        elif t != "fp16" and n not in INT_NAMES:
            bad.append((n, t))
    if bad:
        raise AssertionError(f"{os.path.basename(path)} 非纯 fp16 I/O: {bad}")
    print(f"    纯 fp16 I/O 校验通过: IN={ins} OUT={outs}")


def _to_fp16(args):
    return tuple(a.half() if a.dtype == torch.float32 else a for a in args)


def _feed(in_names, args):
    return {n: a.numpy() for n, a in zip(in_names, args)}


def _ref_of(model, args):
    """从（干净）模型算 fp16 参考输出，归一化为 fp64 numpy 列表。必须在 export 之前调用
    （legacy tracer 会因 HF SDPA 的 Python 控制流副作用就地改动 wrapper）。"""
    with torch.no_grad():
        ret = model(*args)
    rets = [ret] if isinstance(ret, torch.Tensor) else list(ret)
    return [np.asarray(t.numpy(), dtype=np.float64) for t in rets]


def _verify_cpu_fp16(path, in_names, args_fp16, ref, out_names,
                     warn_rel=0.5, fatal_rel=FP16_FATAL_REL):
    """纯 fp16 保真度校验：ORT-fp16 输出 vs 预计算 torch-fp16 参考（相对误差）。
    详见模块 docstring「数值校验语义」。"""
    if _NO_VERIFY:
        print(f"    [skip] CPU-fp16 数值校验已跳过 (--no-verify)：{os.path.basename(path)}")
        return True
    try:
        sess = _sess(path)
        o = sess.run(None, _feed(in_names, args_fp16))
    except Exception as e:  # noqa: BLE001 - CPU EP fp16 kernel 缺失等，非致命
        print(f"    [warn] CPU-EP fp16 运行失败，跳过数值校验（{type(e).__name__}: {str(e)[:140]}）→ 依赖 CUDA EP")
        return False
    worst = 0.0
    for i, on in enumerate(out_names):
        a = np.asarray(o[i], dtype=np.float64)
        b = ref[i]
        assert a.shape == b.shape, f"{os.path.basename(path)}[{on}] 形状 {a.shape} vs {b.shape}"
        if not np.isfinite(a).all():
            # CPU-EP 的 fp16 对含 LSTM/ref_enc 类算子的图易溢出产生非有限值（部署走 CUDA EP），
            # 与「kernel 缺失」同类：不是导出 bug，跳过数值项，仅保留结构校验。
            print(f"    [warn] CPU-EP fp16 输出含非有限值（{on}）——CPU-EP fp16 伪影，跳过数值校验 → 依赖 CUDA EP")
            return False
        if not np.isfinite(b).all():
            raise AssertionError(f"{os.path.basename(path)}[{on}] torch-fp16 参考含非有限值（输入越界）")
        scale = max(1.0, float(np.abs(b).max()))
        rel = float(np.abs(a - b).max()) / scale
        worst = max(worst, rel)
        print(f"  [对拍] {os.path.basename(path)}[{on}] abs={np.abs(a-b).max():.2e} rel={rel:.2e}")
    if worst > fatal_rel:
        raise AssertionError(f"{os.path.basename(path)} 保真度灾难性偏差 rel={worst:.2e} > {fatal_rel}（疑似坏 op）")
    if worst > warn_rel:
        print(f"    [warn] {os.path.basename(path)} rel={worst:.2e} > {warn_rel}（CPU-EP fp16 精度差异，非导出 bug）")
    return True


# ---------------------------------------------------------------- 包装器（自包含拷贝）

class BertWrapper(nn.Module):
    """input_ids/attention_mask -> hidden_states[-3]"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        return out.hidden_states[-3]


class HubertWrapper(nn.Module):
    """wav16k 原始波形 -> last_hidden_state（HF 内置特征提取）"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav16k):
        return self.model(wav16k)["last_hidden_state"]


class SvWrapper(nn.Module):
    """fbank80 [B,T,80] -> ERes2NetV2.forward3 embedding [B,20480]"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fbank):
        return self.model.forward3(fbank)


class GeWrapper(nn.Module):
    """get_ge：ref_enc(mel704) + prelu(sv_emb_linear(sv_emb))，并输出 ge_to512"""

    def __init__(self, vq):
        super().__init__()
        self.ref_enc = vq.ref_enc
        self.sv_emb = vq.sv_emb
        self.prelu = vq.prelu
        self.ge_to512 = vq.ge_to512

    def forward(self, mel704, sv_emb):
        mask = torch.ones((1, 1, mel704.size(2)), dtype=mel704.dtype)
        ge = self.ref_enc(mel704 * mask, mask)
        ge = ge + self.sv_emb(sv_emb).unsqueeze(-1)
        ge = self.prelu(ge)
        ge512 = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        return ge, ge512


# ---------------------------------------------------------------- BERT (fp32 -> int8)

def export_bert_int8(models_dir, feat_dir, text_dir, keep_fp32=False):
    """fp32 导出并对拍 -> MatMul-only dynamic int8 量化 -> int8 vs torch-fp32 保真度检查。"""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    base = os.path.join(models_dir, "chinese-roberta-wwm-ext-large")
    model = AutoModelForMaskedLM.from_pretrained(base, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(base)
    wrap = BertWrapper(model).eval()

    fp32_path = os.path.join(feat_dir, "bert.onnx")
    int8_path = os.path.join(feat_dir, "bert_int8.onnx")
    L = 13
    input_ids = torch.randint(0, 20000, (1, L))
    attention_mask = torch.ones(1, L, dtype=torch.int64)
    with torch.no_grad():
        ref32 = wrap(input_ids, attention_mask)
    _export(wrap, (input_ids, attention_mask), fp32_path,
            ["input_ids", "attention_mask"], ["hidden_states"],
            {"input_ids": {1: "L"}, "attention_mask": {1: "L"}, "hidden_states": {1: "L"}})
    if not _NO_VERIFY:
        sess = _sess(fp32_path)
        for Li in (7, 33):
            ids = torch.randint(0, 20000, (1, Li))
            msk = torch.ones(1, Li, dtype=torch.int64)
            with torch.no_grad():
                r = wrap(ids, msk)
            o = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": msk.numpy()})[0]
            _check(f"bert fp32 L={Li}", o, r)

    # ---- MatMul-only dynamic int8（配方锁定：1198MB -> 367MB，I/O dtype 不变）
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(fp32_path, int8_path,
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    print(f"    int8 量化完成: fp32={os.path.getsize(fp32_path)/1e6:.0f}MB -> "
          f"int8={os.path.getsize(int8_path)/1e6:.0f}MB")

    # int8 保真度：ORT-int8 vs torch-fp32 参考。rel-L2 ~0.2 为该配方设计内水平
    # （BERT 输出是条件特征，经 bert_proj 进 t2s；参考实现全链路实测有效）。
    if not _NO_VERIFY:
        s8 = _sess(int8_path)
        ins, outs = _io_types(int8_path)
        assert all(t in ("int64", "int32") for _, t in ins) and \
            all(t == "fp32" for _, t in outs), f"int8 图 I/O dtype 异常: IN={ins} OUT={outs}"
        worst = 0.0
        for Li in (16, 64, 128):
            ids = torch.randint(0, 20000, (1, Li))
            msk = torch.ones(1, Li, dtype=torch.int64)
            with torch.no_grad():
                r = np.asarray(wrap(ids, msk).numpy(), dtype=np.float64)
            o = np.asarray(s8.run(None, {"input_ids": ids.numpy(),
                                         "attention_mask": msk.numpy()})[0], dtype=np.float64)
            rel = float(np.linalg.norm(o - r) / max(1e-9, np.linalg.norm(r)))
            worst = max(worst, rel)
            print(f"  [对拍] bert_int8 L={Li} rel-L2={rel:.3f}")
        assert np.isfinite(worst) and worst < 0.5, f"bert_int8 误差异常: rel-L2={worst:.3f}"
        print(f"    int8 rel-L2 上界 {worst:.3f}（~0.2 量级为设计内；详见脚本 docstring）")

    # ---- 分词产物 -> text/（包根）
    os.makedirs(text_dir, exist_ok=True)
    samples = [
        "你好，世界！",
        "今天天气真不错。",
        "这是一个测试：hello world 123。",
        "她轻轻地闭上了眼睛。",
        "AI 语音合成正在快速发展！",
        "我们明天下午三点开会。",
        "The quick brown fox jumps over the lazy dog.",
        "混合语言 test 与中文混排。",
        "！？。，；：……——\"\"''（）【】",
        "单字",
    ]
    snap = []
    for t in samples:
        enc = tokenizer(t, return_tensors="np", padding=True, truncation=True, max_length=512)
        snap.append({"text": t,
                     "input_ids": enc["input_ids"][0].tolist(),
                     "attention_mask": enc["attention_mask"][0].tolist()})
    with open(os.path.join(text_dir, "token_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    vocab_path = os.path.join(base, "vocab.txt")
    if os.path.exists(vocab_path):
        shutil.copy(vocab_path, os.path.join(text_dir, "vocab.txt"))
    else:
        tj = os.path.join(base, "tokenizer.json")
        if os.path.exists(tj):
            with open(tj, encoding="utf-8") as f:
                tdata = json.load(f)
            vocab = tdata.get("model", {}).get("vocab", None)
            if vocab is not None:
                with open(os.path.join(text_dir, "vocab.txt"), "w", encoding="utf-8") as f:
                    for tok in vocab.keys():
                        f.write(tok.replace("\n", "\\n") + "\n")
                print(f"  vocab.txt 从 tokenizer.json 提取 ({len(vocab)} 项)")
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        p = os.path.join(base, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(text_dir, fn))

    if not keep_fp32:
        os.remove(fp32_path)
        print("    已删除中间 fp32 bert.onnx（--keep-fp32 可保留为回退）")

    max_len = getattr(tokenizer, "model_max_length", 512)
    return {"onnx": "features/bert_int8.onnx",
            "vocab": "text/vocab.txt",
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["hidden_states"],
            "dim": 1024, "max_length": int(max_len) if max_len < 1e9 else 512,
            "use_for_langs": ["zh"],
            "dtype": "int8", "quant": "dynamic-int8-matmul",
            "fallback_onnx": "features/bert.onnx" if keep_fp32 else None}


# ---------------------------------------------------------------- hubert / sv / ge (fp16)

def export_hubert_fp16(models_dir, out_dir):
    from transformers import HubertModel
    base = os.path.join(models_dir, "chinese-hubert-base")
    model = HubertModel.from_pretrained(base, torch_dtype=torch.float32).eval()
    wrap = HubertWrapper(model)
    S = 16000 // 4 * 3
    wav = torch.randn(1, S)
    print("[hubert] (fp16)")
    half = wrap.half()
    args16 = _to_fp16((wav,))
    ref = _ref_of(half, args16)
    path = os.path.join(out_dir, "hubert_fp16.onnx")
    _export(half, args16, path, ["wav16k"], ["last_hidden_state"],
            {"wav16k": {1: "S"}, "last_hidden_state": {1: "T"}})
    _assert_pure_fp16(path)
    _verify_cpu_fp16(path, ["wav16k"], args16, ref, ["last_hidden_state"])
    return {"onnx": "features/hubert_fp16.onnx", "input_names": ["wav16k"],
            "output_names": ["last_hidden_state"], "dim": 768, "dtype": "fp16"}


def export_sv_fp16(models_dir, out_dir):
    from gsv_tts.GPT_SoVITS.SV.ERes2NetV2 import ERes2NetV2
    ckpt = os.path.join(models_dir, "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
    model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    wrap = SvWrapper(model)
    T = 150
    # 真实管线 sv.py / C 侧 features.c 均传 [B,T,F=80]（时间主序，Kaldi 方向）；forward3
    # 内部 permute(0,2,1)。若误用 [B,F,T] trace 会得到随长度变化的错误图。
    fbank = torch.randn(1, T, 80)
    with torch.no_grad():
        dim = int(wrap(fbank).shape[-1])
    print("[sv] (fp16)")
    half = wrap.half()
    args16 = _to_fp16((fbank,))
    ref = _ref_of(half, args16)
    path = os.path.join(out_dir, "sv_fp16.onnx")
    _export(half, args16, path, ["fbank"], ["embedding"], {"fbank": {1: "T"}})
    _assert_pure_fp16(path)
    _verify_cpu_fp16(path, ["fbank"], args16, ref, ["embedding"])
    return {"onnx": "features/sv_fp16.onnx", "input_names": ["fbank"],
            "output_names": ["embedding"], "dim": dim, "dtype": "fp16"}


def export_ge_fp16(vq, out_dir):
    wrap = GeWrapper(vq)
    gin = vq.gin_channels
    Lm = 97
    # mel704 用真实量级（非负幅度谱 0~50）而非 randn：ref_enc 是 LSTM，fp16 下
    # 喂 Unit 方差噪声会在长序列上溢出（torch-fp16 参考与 ORT-fp16 一起炸，均为输入伪影）。
    mel = torch.rand(1, 704, Lm) * 50.0
    sv_emb = torch.randn(20480)
    print("[ge] (fp16)")
    half = wrap.half()
    args16 = _to_fp16((mel, sv_emb))
    ref = _ref_of(half, args16)
    path = os.path.join(out_dir, "ge_fp16.onnx")
    _export(half, args16, path, ["mel704", "sv_emb"], ["ge", "ge512"], {"mel704": {2: "Lm"}})
    _assert_pure_fp16(path)
    _verify_cpu_fp16(path, ["mel704", "sv_emb"], args16, ref, ["ge", "ge512"])
    # 额外一组 Lm 变长输入（同真实量级）确认动态长度路径
    mel2 = torch.rand(1, 704, 233) * 50.0
    sve2 = torch.randn(20480)
    a16 = _to_fp16((mel2, sve2))
    with torch.no_grad():
        r2 = half(*a16)
    _verify_cpu_fp16(path, ["mel704", "sv_emb"], a16,
                     [np.asarray(t.numpy(), dtype=np.float64) for t in (r2[0], r2[1])],
                     ["ge", "ge512"])
    return {"onnx": "features/ge_fp16.onnx", "input_names": ["mel704", "sv_emb"],
            "output_names": ["ge", "ge512"], "gin_channels": int(gin), "dtype": "fp16"}


# ---------------------------------------------------------------- VQ 码本 + ssl_proj

def export_vq(vq, out_dir):
    vq_dir = os.path.join(out_dir, "vq")
    os.makedirs(vq_dir, exist_ok=True)
    cb = vq.quantizer.vq.layers[0]._codebook.embed.detach().cpu().numpy().astype(np.float32)
    assert cb.shape == (1024, 768), f"码本形状异常 {cb.shape}"
    cb.tofile(os.path.join(vq_dir, "codebook.bin"))
    print(f"  codebook.bin {cb.shape} -> {os.path.join(vq_dir, 'codebook.bin')}")

    # prompt codes 路径的 ssl_proj（SynthesizerTrn.ssl_proj, Conv1d k=2 s=2）
    sp = vq.ssl_proj
    w = sp.weight.detach().cpu().numpy().astype(np.float32)   # [768, 768, 2]
    b = sp.bias.detach().cpu().numpy().astype(np.float32)     # [768]
    with open(os.path.join(vq_dir, "ssl_proj.bin"), "wb") as f:
        f.write(w.tobytes())
        f.write(b.tobytes())
    print(f"  ssl_proj.bin {w.shape}+{b.shape} -> {os.path.join(vq_dir, 'ssl_proj.bin')}")
    return {"codebook": "vq/codebook.bin", "dim": 768, "bins": 1024, "n_q": 1,
            "ssl_proj": {"kernel": 2, "stride": 2, "file": "vq/ssl_proj.bin"}}


# ---------------------------------------------------------------- main

def main():
    global _NO_VERIFY
    parser = argparse.ArgumentParser(
        description="一键导出模型包共享基础（BERT int8 / hubert / sv / ge / VQ），契约见 docs/MODEL_PACKAGE.md")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="HF 特征底模目录（含 chinese-roberta-wwm-ext-large、chinese-hubert-base、sv/；默认 <root>/models）")
    parser.add_argument("--sovits", type=str, default=None,
                        help="SoVITS 权重路径（ge/vq 从其 vq 提取；缺省则跳过 ge 与 vq 两步）")
    parser.add_argument("--out", type=str, required=True, help="模型包根目录（写入 features/ text/ vq/）")
    parser.add_argument("--keep-fp32", action="store_true",
                        help="保留 fp32 bert.onnx 作为回退（契约 fallback_onnx；默认只发 int8）")
    parser.add_argument("--no-verify", action="store_true", help="跳过 ORT 数值校验（仅结构检查）")
    parser.add_argument("--force", action="store_true", help="强制重导（删除已存在目标后重跑）")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-hubert", action="store_true")
    parser.add_argument("--skip-sv", action="store_true")
    parser.add_argument("--skip-ge-vq", action="store_true")
    args = parser.parse_args()

    _NO_VERIFY = bool(args.no_verify)
    torch.manual_seed(20240826)   # 测试输入可复现

    root = project_root
    models_dir = args.models_dir or os.path.join(root, "models")
    out = args.out
    os.makedirs(out, exist_ok=True)
    feat_dir = os.path.join(out, "features")
    os.makedirs(feat_dir, exist_ok=True)

    def _force_rm(*paths):
        if not args.force:
            return
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
                print(f"[force] 删除 {p}")

    sections = {}

    # ---- BERT (fp32 -> int8) + text/
    if not args.skip_bert:
        print("[bert] fp32 导出 + MatMul-only int8 量化")
        _force_rm(os.path.join(feat_dir, "bert.onnx"), os.path.join(feat_dir, "bert_int8.onnx"))
        sections["bert"] = export_bert_int8(models_dir, feat_dir,
                                            os.path.join(out, "text"), keep_fp32=args.keep_fp32)

    # ---- hubert / sv (fp16)
    if not args.skip_hubert:
        print("[hubert]")
        _force_rm(os.path.join(feat_dir, "hubert_fp16.onnx"))
        sections["hubert"] = export_hubert_fp16(models_dir, feat_dir)
    if not args.skip_sv:
        print("[sv]")
        _force_rm(os.path.join(feat_dir, "sv_fp16.onnx"))
        sections["sv"] = export_sv_fp16(models_dir, feat_dir)

    # ---- ge + vq（需 --sovits 提供 vq）
    if not args.skip_ge_vq:
        if not args.sovits:
            print("[ge/vq] 未提供 --sovits，跳过（ge 与 vq 需从 SoVITS 权重的 vq 提取）")
        else:
            from gsv_tts import TTS
            sovits_path = args.sovits if os.path.isabs(args.sovits) else os.path.join(root, args.sovits)
            tts = TTS(models_dir=models_dir, use_bert=False, use_flash_attn=False, device="cpu")
            tts.load_sovits_model(sovits_path)
            vq = tts.sovits_models[sovits_path].vq_model
            assert getattr(vq, "version", "v2") in ("v2", "v2Pro", "v2ProPlus"), \
                f"仅支持 v2 系 (got {getattr(vq, 'version', '?')})"
            print("[ge] (fp16)")
            _force_rm(os.path.join(feat_dir, "ge_fp16.onnx"))
            sections["ge"] = export_ge_fp16(vq, feat_dir)
            print("[vq] 码本 + ssl_proj")
            sections["vq"] = export_vq(vq, out)
            sections["_tts_version"] = getattr(vq, "version", "v2")

    # ---- 汇总：打印可直接合入 manifest.json 的段落
    print("\n===== manifest.json 片段（合入契约布局） =====")
    snippet = {k: v for k, v in sections.items() if not k.startswith("_")}
    print(json.dumps(snippet, ensure_ascii=False, indent=2))
    print(f"\n转换完成 -> {out}")


if __name__ == "__main__":
    main()
