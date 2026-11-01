#!/usr/bin/env python
"""convert_sovits.py - 一键把 SoVITS / GPT(t2s) 权重转成我们的 ONNX 部署格式。

产物（写入 --out，布局见 docs/MODEL_PACKAGE.md 模型包契约）:
    gpt/<gpt_stem>/t2s_prefill_fp16.onnx, t2s_step_fp16.onnx     (GPT，仅 fp16)
    voices/<name>/enc_p_fp32.onnx    (--no-fp32 时跳过；CPU 用)
    voices/<name>/enc_p_fp16.onnx    (CUDA 用)
    voices/<name>/flowdec_fp32.onnx  (--no-fp32 时跳过；CPU 用)
    voices/<name>/flowdec_fp16.onnx  (CUDA 用)

设计要点:
- enc_p = encp_a + encp_b 合并（编码器链 ssl_proj→encoder_ssl→encoder_text→MRTE→encoder2 + proj）。
  速度插值/噪声留在 C 侧：proj 是 1x1 conv（逐位置线性），与线性插值可交换，故把插值移到
  enc_p 之后（作用 m/logs/y_mask）数值等价 torch，且免去图内动态 Resize。
- t2s / flowdec / enc_p 的 ONNX 图与执行设备无关；--device 只决定"在哪台设备上 trace +
  数值校验"，优先 GPU（更快，顺带验证 CUDA EP 可跑），显存不足或不可用则自动回退 CPU。
- fp32 版必须先导出（.half() 会就地改共享子模块），再 .half() 导 fp16。
- 参考输出在 torch.onnx.export **之前**从干净模型算出（legacy tracer 会因 HF SDPA 的 Python
  控制流副作用就地改动 wrapper，导出后再跑同一 wrapper 会得到被污染结果）。

用法:
    runtime/python.exe tools/convert_sovits.py \
        --gpt models/s1v3.ckpt \
        --sovits SoVITS_weights_v2ProPlus/<voice>.pth \
        [--sovits ...更多音色...] \
        --out <staging_dir> \
        [--no-fp32] [--device auto|cuda|cpu] \
        [--ly-max 1024] [--lt-max 512] [--no-verify]

    # 只转 GPT:   runtime/python.exe tools/convert_sovits.py --gpt models/s1v3.ckpt --out <dir>
    # 只转音色:   runtime/python.exe tools/convert_sovits.py --sovits xxx.pth --out <dir> --no-fp32
"""
import argparse
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (project_root, os.path.join(project_root, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import math
import time

import numpy as np
import torch
from torch import nn

import onnxruntime as ort
import onnx

OPSET = 17

ET = {1: "fp32", 10: "fp16", 7: "int64", 6: "int32", 9: "bool", 3: "int8"}
INT_NAMES = {"text_ids", "x_ids", "y_ids", "token_id", "pos_index", "idx", "valid_len",
             "input_ids", "attention_mask"}

# --no-verify：跳过 ORT 数值运行（仅保留结构检查）。
_NO_VERIFY = False


# ---------------------------------------------------------------- 设备选择
def pick_device(mode):
    """返回 (torch_device, ort_providers)。优先 GPU；显存不足/不可用自动回退 CPU。"""
    want_cuda = mode == "cuda" or (mode == "auto" and torch.cuda.is_available())
    if not want_cuda:
        return torch.device("cpu"), ["CPUExecutionProvider"]
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        free = 0
    if free < (2 << 30):   # 需要 ~2GB 空闲显存才值得上 GPU（模型 fp16 + trace 激活）
        print(f"[device] CUDA 空闲 {free/1e9:.2f}GB < 2GB → 回退 CPU")
        return torch.device("cpu"), ["CPUExecutionProvider"]
    provs = ort.get_available_providers()
    use_cuda_ep = "CUDAExecutionProvider" in provs
    print(f"[device] 使用 CUDA（空闲 {free/1e9:.2f}GB）；ORT EP={'CUDA+CPU' if use_cuda_ep else '仅 CPU(ORT 无 CUDA EP，校验走 CPU)'}")
    ort_provs = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda_ep
                 else ["CPUExecutionProvider"])
    return torch.device("cuda"), ort_provs


# ---------------------------------------------------------------- 导出助手
def _export(model, args_tuple, path, input_names, output_names, dynamic_axes):
    print(f"  导出 {os.path.basename(path)} …")
    t0 = time.time()
    torch.onnx.export(
        model, args_tuple, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET, dynamo=False, do_constant_folding=True,
    )
    print(f"    完成 {os.path.getsize(path)/1e6:.0f}MB {time.time()-t0:.1f}s")


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


def _move(args, dev, dt):
    """float 输入搬到 dev 并转 dt；int 输入只搬 dev。"""
    return tuple(a.to(dev).to(dt) if a.is_floating_point() else a.to(dev) for a in args)


def _ref_of(model, args):
    with torch.no_grad():
        ret = model(*args)
    rets = [ret] if isinstance(ret, torch.Tensor) else list(ret)
    return [np.asarray(t.detach().cpu().numpy()) for t in rets]


def _verify(path, in_names, feed, ref, out_names, providers, warn_rel=0.5, fatal_rel=5.0):
    """ORT 输出 vs 预计算 torch 参考（相对误差）。运行失败→warn 跳过；非有限/rel>fatal→assert。"""
    if _NO_VERIFY:
        print(f"    [skip] 数值校验已跳过 (--no-verify)：{os.path.basename(path)}")
        return
    try:
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        sess = ort.InferenceSession(path, so, providers=providers)
        o = sess.run(None, feed)
    except Exception as e:  # noqa: BLE001 - 缺 kernel / OOM 等，非致命
        print(f"    [warn] ORT 运行失败，跳过数值校验（{type(e).__name__}: {str(e)[:140]}）")
        return
    worst = 0.0
    for i, on in enumerate(out_names):
        a = np.asarray(o[i], dtype=np.float64)
        b = ref[i]
        assert a.shape == b.shape, f"{os.path.basename(path)}[{on}] 形状 {a.shape} vs {b.shape}"
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise AssertionError(f"{os.path.basename(path)}[{on}] 含非有限值（fp16 溢出/输入越界）")
        scale = max(1.0, float(np.abs(b).max()))
        d = float(np.abs(a - b).max())
        rel = d / scale
        worst = max(worst, rel)
        print(f"  [对拍] {os.path.basename(path)}[{on}] abs={d:.2e} rel={rel:.2e}")
    if worst > fatal_rel:
        raise AssertionError(f"{os.path.basename(path)} 保真度灾难性偏差 rel={worst:.2e} > {fatal_rel}（疑似坏 op）")
    if worst > warn_rel:
        print(f"    [warn] {os.path.basename(path)} rel={worst:.2e} > {warn_rel}（EP/dtype 精度差异，非导出 bug）")


def _done(*paths):
    return all(os.path.exists(p) and os.path.getsize(p) > 1024 for p in paths)


# ---------------------------------------------------------------- 包装器（自包含拷贝，避免 export_*.py 的 CUDA_VISIBLE_DEVICES="" 副作用）

def _manual_attn(q, k, v, scale, bias=None):
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if bias is not None:
        scores = scores + bias
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class EncPWrapper(nn.Module):
    """enc_p 合并版 = 编码器链（ssl_proj→encoder_ssl→encoder_text→MRTE→encoder2）+ proj。

    输出 (m_p, logs_p, y_mask) @ valid_len。等价于 encp_a(编码器链) ∘ encp_b(proj)。
    """

    def __init__(self, enc_p, out_channels):
        super().__init__()
        self.ssl_proj = enc_p.ssl_proj
        self.encoder_ssl = enc_p.encoder_ssl
        self.text_embedding = enc_p.text_embedding
        self.encoder_text = enc_p.encoder_text
        self.mrte = enc_p.mrte
        self.encoder2 = enc_p.encoder2
        self.proj = enc_p.proj
        self.out_channels = out_channels

    @staticmethod
    def _enc_masked(enc, x, mask):
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        for i in range(enc.n_layers):
            y = enc.attn_layers[i](x, x, attn_mask)
            x = enc.norm_layers_1[i](x + y) * mask
            y = enc.ffn_layers[i](x, mask)
            x = enc.norm_layers_2[i](x + y) * mask
        return x

    def forward(self, quantized, text_ids, ge512, y_mask, t_mask, valid_len):
        y = self.ssl_proj(quantized * y_mask) * y_mask
        y = self._enc_masked(self.encoder_ssl, y, y_mask)
        text = self.text_embedding(text_ids).transpose(1, 2)
        text = self._enc_masked(self.encoder_text, text, t_mask)
        y = self.mrte(y, y_mask, text, t_mask, ge512) * y_mask
        y = self._enc_masked(self.encoder2, y, y_mask)

        y = torch.narrow(y, 2, 0, valid_len)
        mask = torch.narrow(y_mask, 2, 0, valid_len)
        stats = self.proj(y) * mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, mask


class SoVITSv2FlowDecWrapper(nn.Module):
    """z_p -> 音频：flow 逆变换 + HiFi-GAN（纯卷积，动态长度安全）"""

    def __init__(self, vq):
        super().__init__()
        self.flow = vq.flow
        self.dec = vq.dec

    def forward(self, z_p, y_mask, ge):
        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        return self.dec(z * y_mask, g=ge)


class T2SPrefillWrapper(nn.Module):
    """prompt 段一次前向：嵌入 + 全序列注意力，输出首步 logits 与 prompt 段 KV。"""

    def __init__(self, t2s):
        super().__init__()
        self.num_heads = t2s.num_head
        self.head_dim = t2s.model_dim // t2s.num_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.ar_text_embedding = t2s.ar_text_embedding
        self.bert_proj = t2s.bert_proj
        self.ar_audio_embedding = t2s.ar_audio_embedding
        self.blocks = t2s.t2s_transformer.blocks
        self.ar_predict_layer = t2s.ar_predict_layer
        assert t2s.ar_text_position.x_scale == 1.0 and t2s.ar_audio_position.x_scale == 1.0
        pe_t = (t2s.ar_text_position.alpha.detach() * t2s.ar_text_position.pe).detach()
        pe_a = (t2s.ar_audio_position.alpha.detach() * t2s.ar_audio_position.pe).detach()
        self.register_buffer("pe_text", pe_t.to(torch.float32), persistent=False)
        self.register_buffer("pe_audio", pe_a.to(torch.float32), persistent=False)

    def forward(self, x_ids, y_ids, bert, attn_bias):
        Lx = x_ids.shape[1]
        Ly = y_ids.shape[1]
        x = self.ar_text_embedding(x_ids) + self.bert_proj(bert)
        x = x + self.pe_text[:, :Lx]
        y = self.ar_audio_embedding(y_ids)
        y = y + self.pe_audio[:, :Ly]
        h = torch.cat([x, y], dim=1)
        S = h.shape[1]

        ks, vs = [], []
        for blk in self.blocks:
            residual = h
            qkv = blk.qkv(h).view(1, S, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            a = _manual_attn(q, k, v, self.scale, attn_bias)
            a = blk.out_proj(a.transpose(1, 2).reshape(1, S, a.shape[-1] * self.num_heads))
            h = blk.norm1(residual + a)
            residual = h
            h = blk.norm2(residual + blk.mlp(h))
            ks.append(k); vs.append(v)

        logits = self.ar_predict_layer(h[:, -1])
        return logits, torch.stack(ks, 0), torch.stack(vs, 0)


class T2SStepWrapper(nn.Module):
    """单 token 步进：输入完整 KV buffer + valid_len，图内只取有效段。"""

    def __init__(self, t2s, max_len):
        super().__init__()
        self.num_heads = t2s.num_head
        self.head_dim = t2s.model_dim // t2s.num_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.ar_audio_embedding = t2s.ar_audio_embedding
        self.blocks = t2s.t2s_transformer.blocks
        self.ar_predict_layer = t2s.ar_predict_layer
        pe_cache = (t2s.ar_audio_position.alpha.detach() * t2s.ar_audio_position.pe).detach()
        pe_cache = pe_cache.transpose(0, 1).to(torch.float32)[:max_len]
        self.register_buffer("pe_audio_cache", pe_cache, persistent=False)

    def forward(self, token_id, pos_index, idx, past_k, past_v):
        y_emb = self.ar_audio_embedding(token_id)
        h = y_emb + self.pe_audio_cache[pos_index]
        scale = self.scale

        knews, vnews = [], []
        for i, blk in enumerate(self.blocks):
            residual = h
            qkv = blk.qkv(h).view(1, 1, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            pk = torch.index_select(past_k[i], 2, idx)
            pv = torch.index_select(past_v[i], 2, idx)
            a = _manual_attn(q, torch.cat([pk, k], 2), torch.cat([pv, v], 2), scale)
            a = blk.out_proj(a.transpose(1, 2).reshape(1, 1, a.shape[-1] * self.num_heads))
            h = blk.norm1(residual + a)
            residual = h
            h = blk.norm2(residual + blk.mlp(h))
            knews.append(k); vnews.append(v)

        logits = self.ar_predict_layer(h[:, -1])
        return logits, torch.stack(knews, 0), torch.stack(vnews, 0)


# ---------------------------------------------------------------- GPT (t2s，纯 fp16)

def export_gpt(tts, gpt_path, out_dir, dev, providers):
    t2s = tts.gpt_models[gpt_path].t2s_model
    t2s.eval()
    max_len = max(seq for b, seq in tts.tts_config.gpt_cache if b == 1)
    L, H, D = t2s.num_layers, t2s.num_head, t2s.model_dim
    Lx, Ly = 37, 94
    S = Lx + Ly

    prefill = T2SPrefillWrapper(t2s).eval().to(dev)
    x_ids = torch.randint(0, t2s.phoneme_vocab_size, (1, Lx))
    y_ids = torch.randint(0, t2s.vocab_size, (1, Ly))
    bert = torch.randn(1, Lx, 1024)
    mask = torch.zeros(S, S)
    mask[:Lx, :Lx] = 1; mask[Lx:, :Lx] = 1; mask[Lx:, Lx:] = torch.tril(torch.ones(Ly, Ly))
    bias = torch.where(mask > 0, 0.0, -1e9).view(1, 1, S, S)
    args_p = (x_ids, y_ids, bert, bias)

    pf = os.path.join(out_dir, "t2s_prefill_fp16.onnx")
    if _done(pf):
        print(f"[gpt] t2s_prefill_fp16 已存在，跳过")
    else:
        prefill.half()                                   # 就地（共享 t2s 子模块）
        a16 = _move(args_p, dev, torch.float16)
        ref = _ref_of(prefill, a16)                      # 干净参考（export 前）
        _export(prefill, a16, pf, ["x_ids", "y_ids", "bert", "attn_bias"],
                ["logits", "present_k", "present_v"],
                {"x_ids": {1: "Lx"}, "y_ids": {1: "Ly"}, "bert": {1: "Lx"},
                 "attn_bias": {2: "S", 3: "S"}, "present_k": {3: "S"}, "present_v": {3: "S"}})
        _assert_pure_fp16(pf)
        _verify(pf, ["x_ids", "y_ids", "bert", "attn_bias"],
                {n: a.cpu().numpy() for n, a in zip(["x_ids", "y_ids", "bert", "attn_bias"], a16)},
                ref, ["logits", "present_k", "present_v"], providers)

    step = T2SStepWrapper(t2s, max_len).eval().to(dev)
    token = torch.tensor([[123]]); pos = torch.tensor([Ly]); idx = torch.arange(S)
    past_k = torch.randn(L, 1, H, max_len, D // H)
    past_v = torch.randn(L, 1, H, max_len, D // H)
    args_s = (token, pos, idx, past_k, past_v)

    sf = os.path.join(out_dir, "t2s_step_fp16.onnx")
    if _done(sf):
        print(f"[gpt] t2s_step_fp16 已存在，跳过")
    else:
        step.half()
        a16s = _move(args_s, dev, torch.float16)
        refs = _ref_of(step, a16s)
        _export(step, a16s, sf, ["token_id", "pos_index", "idx", "past_k", "past_v"],
                ["logits", "k_new", "v_new"], {"idx": {0: "T"}})
        _assert_pure_fp16(sf)
        _verify(sf, ["token_id", "pos_index", "idx", "past_k", "past_v"],
                {n: a.cpu().numpy() for n, a in zip(["token_id", "pos_index", "idx", "past_k", "past_v"], a16s)},
                refs, ["logits", "k_new", "v_new"], providers)

    return {"num_layers": int(L), "num_heads": int(H), "model_dim": int(D),
            "max_len": int(max_len), "vocab_size": int(t2s.vocab_size), "eos": int(t2s.EOS)}


# ---------------------------------------------------------------- SoVITS 音色 (enc_p 合并 + flowdec)

def export_voice(vq, name, out_dir, dev, providers, want_fp32, ly_max=1024, lt_max=512):
    vdir = os.path.join(out_dir, "voices", name)
    os.makedirs(vdir, exist_ok=True)

    enc_p = vq.enc_p
    inter = vq.inter_channels
    gin = vq.gin_channels
    n_symbols = enc_p.text_embedding.num_embeddings

    # ---- 输入（CPU 上构造，导出时再搬设备/转 dtype）----
    quantized = torch.randn(1, 768, ly_max)
    text_ids = torch.randint(0, n_symbols, (1, lt_max))
    ge512 = torch.randn(1, 512, 1)
    y_mask = torch.ones(1, 1, ly_max); t_mask = torch.ones(1, 1, lt_max)
    valid = torch.tensor([2 * 37], dtype=torch.int64)
    encp_args = (quantized, text_ids, ge512, y_mask, t_mask, valid)
    in_encp = ["quantized", "text_ids", "ge512", "y_mask", "t_mask", "valid_len"]
    out_encp = ["m_p", "logs_p", "y_mask_out"]

    Lf = 128
    z_p = torch.randn(1, inter, Lf); ym_f = torch.ones(1, 1, Lf); ge1 = torch.randn(1, gin, 1)
    flow_args = (z_p, ym_f, ge1)
    in_flow = ["z_p", "y_mask", "ge"]; out_flow = ["audio"]
    dyn_flow = {"z_p": {2: "Lz"}, "y_mask": {2: "Lz"}, "ge": {2: "Lz"}}

    encp_fp32 = os.path.join(vdir, "enc_p_fp32.onnx")
    encp_fp16 = os.path.join(vdir, "enc_p_fp16.onnx")
    flow_fp32 = os.path.join(vdir, "flowdec_fp32.onnx")
    flow_fp16 = os.path.join(vdir, "flowdec_fp16.onnx")

    wrap_enc = EncPWrapper(enc_p, inter).eval().to(dev)
    wrap_flow = SoVITSv2FlowDecWrapper(vq).eval().to(dev)

    # ---- fp32（先导出，.half() 会就地改共享子模块）----
    if want_fp32:
        if not _done(encp_fp32):
            a = _move(encp_args, dev, torch.float32)
            ref = _ref_of(wrap_enc, a)
            _export(wrap_enc, a, encp_fp32, in_encp, out_encp, {})
            _verify(encp_fp32, in_encp, {n: x.cpu().numpy() for n, x in zip(in_encp, a)}, ref, out_encp, providers)
        else:
            print(f"[{name}] enc_p_fp32 已存在，跳过")
        if not _done(flow_fp32):
            a = _move(flow_args, dev, torch.float32)
            ref = _ref_of(wrap_flow, a)
            _export(wrap_flow, a, flow_fp32, in_flow, out_flow, dyn_flow)
            _verify(flow_fp32, in_flow, {n: x.cpu().numpy() for n, x in zip(in_flow, a)}, ref, out_flow, providers)
        else:
            print(f"[{name}] flowdec_fp32 已存在，跳过")

    # ---- fp16（就地 half）----
    wrap_enc.half(); wrap_flow.half()
    if not _done(encp_fp16):
        a = _move(encp_args, dev, torch.float16)
        ref = _ref_of(wrap_enc, a)
        _export(wrap_enc, a, encp_fp16, in_encp, out_encp, {})
        _assert_pure_fp16(encp_fp16)
        _verify(encp_fp16, in_encp, {n: x.cpu().numpy() for n, x in zip(in_encp, a)}, ref, out_encp, providers)
    else:
        print(f"[{name}] enc_p_fp16 已存在，跳过")
    if not _done(flow_fp16):
        a = _move(flow_args, dev, torch.float16)
        ref = _ref_of(wrap_flow, a)
        _export(wrap_flow, a, flow_fp16, in_flow, out_flow, dyn_flow)
        _assert_pure_fp16(flow_fp16)
        _verify(flow_fp16, in_flow, {n: x.cpu().numpy() for n, x in zip(in_flow, a)}, ref, out_flow, providers)
    else:
        print(f"[{name}] flowdec_fp16 已存在，跳过")

    return {"inter_channels": int(inter), "hidden_channels": int(enc_p.hidden_channels),
            "filter_channels": int(enc_p.filter_channels), "n_heads": int(enc_p.n_heads),
            "n_layers": int(enc_p.n_layers), "kernel_size": int(enc_p.kernel_size),
            "gin_channels": int(gin), "enc_p": {"ly_max": int(ly_max), "lt_max": int(lt_max)},
            "samples_per_frame": int(vq.samples_per_frame)}


# ---------------------------------------------------------------- main

def main():
    global _NO_VERIFY
    parser = argparse.ArgumentParser(description="一键转换 SoVITS / GPT(t2s) 为我们的 ONNX 部署格式")
    parser.add_argument("--gpt", type=str, default=None, help="GPT(t2s) 权重路径，如 models/s1v3.ckpt")
    parser.add_argument("--sovits", type=str, action="append", default=None,
                        help="SoVITS 权重路径（可多次），如 SoVITS_weights_v2ProPlus/<voice>.pth")
    parser.add_argument("--out", type=str, required=True, help="输出目录（staging）")
    parser.add_argument("--models-dir", type=str, default=None, help="预训练模型目录（默认 <root>/models）")
    parser.add_argument("--no-fp32", action="store_true", help="不导出 fp32 版 enc_p/flowdec（仅 GPU 部署时）")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="trace+校验设备，默认 auto（优先 GPU，不足回退 CPU）")
    parser.add_argument("--ly-max", type=int, default=1024)
    parser.add_argument("--lt-max", type=int, default=512)
    parser.add_argument("--no-verify", action="store_true", help="跳过 ORT 数值校验（仅结构检查）")
    parser.add_argument("--force", action="store_true", help="强制重新导出（删除目标已存在文件后重跑+重校验）")
    args = parser.parse_args()

    if not args.gpt and not args.sovits:
        parser.error("至少指定 --gpt 或 --sovits")
    _NO_VERIFY = bool(args.no_verify)
    want_fp32 = not args.no_fp32

    torch.manual_seed(20240826)   # 测试输入可复现，且避免无界 randn 触发 fp16 溢出
    dev, providers = pick_device(args.device)

    root = project_root
    models_dir = args.models_dir or os.path.join(root, "models")
    out = args.out
    os.makedirs(out, exist_ok=True)

    if args.force:
        import shutil
        targets = []
        if args.gpt:
            gp = args.gpt if os.path.isabs(args.gpt) else os.path.join(root, args.gpt)
            targets.append(os.path.join(out, "gpt", os.path.splitext(os.path.basename(gp))[0]))
        for sp in (args.sovits or []):
            s = sp if os.path.isabs(sp) else os.path.join(root, sp)
            targets.append(os.path.join(out, "voices", os.path.splitext(os.path.basename(s))[0]))
        for d in targets:
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"[force] 删除 {d}")

    from gsv_tts import TTS
    # 权重统一在 CPU 载入（避免 CUDA graph cache 预占显存），导出时再按需搬设备。
    tts = TTS(models_dir=models_dir, use_bert=False, use_flash_attn=False, device="cpu")

    if args.gpt:
        gpt_path = args.gpt if os.path.isabs(args.gpt) else os.path.join(root, args.gpt)
        stem = os.path.splitext(os.path.basename(gpt_path))[0]
        gpt_dir = os.path.join(out, "gpt", stem)
        os.makedirs(gpt_dir, exist_ok=True)
        tts.load_gpt_model(gpt_path)
        print(f"[gpt] {stem} (device={dev})")
        export_gpt(tts, gpt_path, gpt_dir, dev, providers)

    for sp in (args.sovits or []):
        sovits_path = sp if os.path.isabs(sp) else os.path.join(root, sp)
        name = os.path.splitext(os.path.basename(sovits_path))[0]
        tts.load_sovits_model(sovits_path)
        vq = tts.sovits_models[sovits_path].vq_model
        assert getattr(vq, "version", "v2") in ("v2", "v2Pro", "v2ProPlus"), \
            f"{name}: 仅支持 v2 系 (got {getattr(vq, 'version', '?')})"
        print(f"[sovits] {name} (device={dev}, fp32={'on' if want_fp32 else 'off'})")
        export_voice(vq, name, out, dev, providers, want_fp32, args.ly_max, args.lt_max)

    print(f"\n转换完成 -> {out}")


if __name__ == "__main__":
    main()
