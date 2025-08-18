#!/usr/bin/env python3
"""
convert_glm45v_mmproj.py  •  UPDATED FOR HEADER‑ALIGNED ALIASES

Usage (unchanged):
    python3 convert_glm45v_mmproj.py \
        --model THUDM/GLM-4.5V \
        --outfile /models/GLM-4.5v/model.mmproj.gguf \
        --dtype f16

Key changes
===========
* **All tensor names now match the canonical macros in *clip.h*.**  This
  eliminates the former snake‑case variants (`ln_1`, `ln_post`, etc.) and
  removes non‑standard aliases such as `attn_output`.
* **Metadata keys unchanged**, as they already follow the
  `KEY_*` definitions – but we now explicitly emit `clip.%s.layer_norm_epsilon`
  for every encoder (`vision`, `audio`, …) so it lines up with
  `KEY_LAYER_NORM_EPS`.
* **Canonical first – compatibility second.**  For any legacy spellings that
  were previously required by downstream loaders we still write *secondary*
  aliases.  The canonical header form is always written *first* so that gguf
  readers take that as the primary tensor.
* **Positional embedding** is now `v.position_embd.weight` (`TN_POS_EMBD`).
* **Post‑VIT norm** is now `v.post_ln.weight` (`TN_LN_POST`).
* **Block norms** are now `ln1` / `ln2` (no underscores) per `TN_LN_1/2`.

Feel free to diff this with your original script – almost all logic is
unchanged, only the alias strings were touched.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
from gguf import GGUFWriter, VisionProjectorType

# -------------------- helpers --------------------

def _as_numpy(t, dtype: str):
    if t is None:
        return None
    if dtype == "f16":
        t = t.detach().cpu().to(torch.float16)
    elif dtype == "f32":
        t = t.detach().cpu().to(torch.float32)
    else:
        raise ValueError("dtype must be f16 or f32")
    return t.numpy()


def _write_alias(tensors, name: str, arr):
    if arr is not None:
        tensors.append((name, arr))


def _alias_pair(tensors, a: str, b: str, arr):
    if arr is not None:
        tensors.append((a, arr))  # canonical first
        tensors.append((b, arr))


def _split_qkv(w: torch.Tensor, b: torch.Tensor | None, hidden: int):
    """HF packs qkv along out‑dim: [3*hidden, hidden]"""
    assert w.shape[0] == 3 * hidden, f"qkv out={w.shape[0]} != 3*{hidden}"
    wq, wk, wv = torch.split(w, hidden, dim=0)
    if b is None:
        return wq, wk, wv, None, None, None
    bq, bk, bv = torch.split(b, hidden, dim=0)
    return wq, wk, wv, bq, bk, bv

# -------------------- metadata --------------------

def _write_meta_for_loader(gg: GGUFWriter, vis_cfg, text_cfg, image_mean, image_std):
    gg.add_string("general.architecture", "mmproj")
    gg.add_string("mmproj.architecture",  "glm4v_moe")

    # KEY_* macros
    gg.add_clip_projector_type(VisionProjectorType.GLM4VMOE)
    gg.add_vision_use_silu(True)
    gg.add_clip_has_audio_encoder(False)
    gg.add_clip_has_vision_encoder(True)
    # gg.add_string("clip.projector_type", "glm4v_moe")

    # Parameter extraction
    n_embd    = int(vis_cfg.hidden_size)
    n_head    = int(vis_cfg.num_heads)
    n_head_kv = int(getattr(vis_cfg, "num_heads_kv", n_head))
    n_ff      = int(vis_cfg.intermediate_size)
    n_block   = int(vis_cfg.depth)
    img_sz    = int(vis_cfg.image_size)
    patch_sz  = int(vis_cfg.patch_size)
    proj_dim  = int(getattr(vis_cfg, "out_hidden_size", text_cfg.hidden_size))
    eps_glob  = float(getattr(vis_cfg, "rms_norm_eps", 1e-5))
    eps_attn  = float(getattr(vis_cfg, "attn_rms_norm_eps", eps_glob))
    spatial_merge = int(getattr(vis_cfg, "spatial_merge_size", 0))

    # Vision encoder hparams – canonical firsts
    gg.add_uint32("clip.vision.embedding_length",    n_embd)
    gg.add_uint32("clip.vision.block_count",         n_block)
    gg.add_uint32("clip.vision.feed_forward_length", n_ff)

    gg.add_uint32("clip.vision.attention.head_count",     n_head)
    gg.add_uint32("clip.vision.attention.head_count_kv",  n_head_kv)
    if n_head:
        gg.add_uint32("clip.vision.attention.dim_head",   n_embd // n_head)
    gg.add_float32("clip.vision.attention.layer_norm_epsilon", eps_attn)

    gg.add_uint32("clip.vision.projection_dim",      proj_dim)
    gg.add_uint32("clip.vision.image_size",          img_sz)
    gg.add_uint32("clip.vision.patch_size",          patch_sz)
    gg.add_float32("clip.vision.layer_norm_epsilon", eps_glob)
    gg.add_uint32("clip.vision.spatial_merge_size",  spatial_merge)

    # Image stats under both namespaces (KEY_IMAGE_* and legacy)
    image_mean = [float(x) for x in image_mean]
    image_std  = [float(x) for x in image_std]
    gg.add_array("clip.vision.image_mean", image_mean)
    gg.add_array("clip.vision.image_std",  image_std)
    gg.add_array("clip.image_mean", image_mean)  # legacy
    gg.add_array("clip.image_std",  image_std)

# -------------------- export --------------------

def export(model, outfile: str, dtype: str = "f16"):
    vis = model.visual
    txt = model.language_model
    vis_cfg = vis.config
    text_cfg = txt.config

    # Validate backbone assumptions (unchanged)
    assert int(vis_cfg.hidden_size)        == 1536
    assert int(vis_cfg.num_heads)          == 12
    assert int(vis_cfg.depth)              == 24
    assert int(vis_cfg.image_size)         == 336
    assert int(vis_cfg.patch_size)         == 14
    assert int(vis_cfg.spatial_merge_size) == 2
    assert int(vis_cfg.out_hidden_size)    == 4096

    # Image preproc stats
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(model.name_or_path, trust_remote_code=True, use_fast=True)
        image_mean = list(proc.image_mean)
        image_std  = list(proc.image_std)
    except Exception:
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std  = [0.26862954, 0.26130258, 0.27577711]

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    gg = GGUFWriter(outfile, "mmproj")
    _write_meta_for_loader(gg, vis_cfg, text_cfg, image_mean, image_std)

    tensors: list[tuple[str, np.ndarray]] = []

    # ---- (1) Patch embed ----
    pe = vis.patch_embed.proj
    w = pe.weight  # Conv2d: [O, I, H, W]; Conv3d: [O, I, T, H, W]

    # Collapse temporal dim for Conv3d (single-image inputs)
    if w.ndim == 5:
        w2d = w.sum(dim=2)           # [O, I, H, W]
    elif w.ndim == 4:
        w2d = w                       # [O, I, H, W]
    else:
        raise ValueError(f"Unexpected patch_embed weight ndim={w.ndim}")

    # IMPORTANT: gguf writer stores shapes directly as ggml ne[0..3].
    # ggml conv_2d expects weights with ne = [KW, KH, IC, OC].
    # The python gguf writer in llama.cpp flips numpy shape order internally,
    # so to land with ne=[KW,KH,IC,OC] we must pass numpy as [OC,IC,KH,KW].
    # Therefore: DO NOT permute to [W,H,I,O] here; give the writer [O,I,H,W].
    w_out = w2d.contiguous()  # numpy will be [OC,IC,KH,KW]

    # Sanity – these reflect torch dims; gguf_dump should then show 14,14,3,1536
    assert tuple(w_out.shape) == (int(vis_cfg.hidden_size), 3, int(vis_cfg.patch_size), int(vis_cfg.patch_size))

    _write_alias(tensors, "v.patch_embd.weight",      _as_numpy(w_out, dtype))  # TN_PATCH_EMBD (canonical)
    _write_alias(tensors, "v.patch_embeddings.weight", _as_numpy(w_out, dtype))  # legacy

    if hasattr(pe, "bias") and pe.bias is not None:
        arrb = _as_numpy(pe.bias, dtype)
        _write_alias(tensors, "v.patch_embd.bias", arrb)      # TN_PATCH_BIAS
        _write_alias(tensors, "v.patch_bias",       arrb)      # legacy

    # Positional embed – TN_POS_EMBD
    pos_emb = getattr(getattr(vis, "embeddings", None), "position_embedding", None)
    if pos_emb is not None and getattr(pos_emb, "weight", None) is not None:
        pem = _as_numpy(pos_emb.weight, dtype)
        _write_alias(tensors, "v.position_embd.weight", pem)  # canonical
        _write_alias(tensors, "v.positional_embedding",  pem)  # legacy

    # Stem RMS (unchanged)
    if hasattr(vis, "post_conv_layernorm"):
        _write_alias(tensors, "glm4v.post_conv_rmsnorm.weight", _as_numpy(vis.post_conv_layernorm.weight, dtype))

    H = int(vis_cfg.hidden_size)

    # ---- (2) Transformer blocks ----
    for i, blk in enumerate(vis.blocks):
        # Block norms – TN_LN_1 / TN_LN_2
        _alias_pair(tensors, f"v.layers.{i}.ln1.weight", f"v.blk.{i}.ln1.weight", _as_numpy(blk.norm1.weight, dtype))
        _alias_pair(tensors, f"v.layers.{i}.ln2.weight", f"v.blk.{i}.ln2.weight", _as_numpy(blk.norm2.weight, dtype))

        # Attention
        wq, wk, wv, bq, bk, bv = _split_qkv(blk.attn.qkv.weight, blk.attn.qkv.bias, H)
        _alias_pair(tensors, f"v.layers.{i}.attn_q.weight", f"v.blk.{i}.attn_q.weight", _as_numpy(wq, dtype))
        _alias_pair(tensors, f"v.layers.{i}.attn_k.weight", f"v.blk.{i}.attn_k.weight", _as_numpy(wk, dtype))
        _alias_pair(tensors, f"v.layers.{i}.attn_v.weight", f"v.blk.{i}.attn_v.weight", _as_numpy(wv, dtype))

        wo = _as_numpy(blk.attn.proj.weight, dtype)
        _alias_pair(tensors, f"v.layers.{i}.attn_out.weight", f"v.blk.{i}.attn_out.weight", wo)  # TN_ATTN_OUTPUT

        # Biases (optional)
        if bq is not None:
            _alias_pair(tensors, f"v.layers.{i}.attn_q.bias", f"v.blk.{i}.attn_q.bias", _as_numpy(bq, dtype))
        if bk is not None:
            _alias_pair(tensors, f"v.layers.{i}.attn_k.bias", f"v.blk.{i}.attn_k.bias", _as_numpy(bk, dtype))
        if bv is not None:
            _alias_pair(tensors, f"v.layers.{i}.attn_v.bias", f"v.blk.{i}.attn_v.bias", _as_numpy(bv, dtype))
        if getattr(blk.attn.proj, "bias", None) is not None:
            wob = _as_numpy(blk.attn.proj.bias, dtype)
            _alias_pair(tensors, f"v.layers.{i}.attn_out.bias", f"v.blk.{i}.attn_out.bias", wob)

        # MLP (SwiGLU)
        mlp = blk.mlp
        _alias_pair(tensors, f"v.layers.{i}.ffn_gate.weight", f"v.blk.{i}.ffn_gate.weight", _as_numpy(mlp.gate_proj.weight, dtype))
        _alias_pair(tensors, f"v.layers.{i}.ffn_up.weight",   f"v.blk.{i}.ffn_up.weight",   _as_numpy(mlp.up_proj.weight, dtype))
        _alias_pair(tensors, f"v.layers.{i}.ffn_down.weight", f"v.blk.{i}.ffn_down.weight", _as_numpy(mlp.down_proj.weight, dtype))

        if getattr(mlp.gate_proj, "bias", None) is not None:
            _alias_pair(tensors, f"v.layers.{i}.ffn_gate.bias", f"v.blk.{i}.ffn_gate.bias", _as_numpy(mlp.gate_proj.bias, dtype))
        if getattr(mlp.up_proj, "bias", None) is not None:
            _alias_pair(tensors, f"v.layers.{i}.ffn_up.bias",   f"v.blk.{i}.ffn_up.bias", _as_numpy(mlp.up_proj.bias, dtype))
        if getattr(mlp.down_proj, "bias", None) is not None:
            _alias_pair(tensors, f"v.layers.{i}.ffn_down.bias", f"v.blk.{i}.ffn_down.bias", _as_numpy(mlp.down_proj.bias, dtype))

        # GLM‑native names retained for debug (unchanged)
        _write_alias(tensors, f"glm4v.blocks.{i}.norm1.weight", _as_numpy(blk.norm1.weight, dtype))
        _write_alias(tensors, f"glm4v.blocks.{i}.norm2.weight", _as_numpy(blk.norm2.weight, dtype))
        _write_alias(tensors, f"glm4v.blocks.{i}.attn.wq",      _as_numpy(wq, dtype))
        _write_alias(tensors, f"glm4v.blocks.{i}.attn.wk",      _as_numpy(wk, dtype))
        _write_alias(tensors, f"glm4v.blocks.{i}.attn.wv",      _as_numpy(wv, dtype))
        if bq is not None: _write_alias(tensors, f"glm4v.blocks.{i}.attn.bq", _as_numpy(bq, dtype))
        if bk is not None: _write_alias(tensors, f"glm4v.blocks.{i}.attn.bk", _as_numpy(bk, dtype))
        if bv is not None: _write_alias(tensors, f"glm4v.blocks.{i}.attn.bv", _as_numpy(bv, dtype))
        _write_alias(tensors, f"glm4v.blocks.{i}.attn.wo",      wo)
        _write_alias(tensors, f"glm4v.blocks.{i}.mlp.gate",     _as_numpy(mlp.gate_proj.weight, dtype))
        _write_alias(tensors, f"glm4v.blocks.{i}.mlp.up",       _as_numpy(mlp.up_proj.weight, dtype))
        _write_alias(tensors, f"glm4v.blocks.{i}.mlp.down",     _as_numpy(mlp.down_proj.weight, dtype))

    # ---- (3) Post‑ViT norm – TN_LN_POST ----
    if hasattr(vis, "post_layernorm"):
        arr = _as_numpy(vis.post_layernorm.weight, dtype)
        _write_alias(tensors, "v.post_ln.weight", arr)  # canonical
        _write_alias(tensors, "glm4v.post_rmsnorm.weight", arr)  # legacy

    # ---- (4) Downsample & Merger (unchanged) ----
    if hasattr(vis, "downsample"):
        ds = vis.downsample
        ds_w = ds.weight.contiguous()
        _write_alias(tensors, "glm4v.downsample.weight", _as_numpy(ds_w, dtype))
        if getattr(ds, "bias", None) is not None:
            b = _as_numpy(ds.bias, dtype)                  # [4096]
            b4 = b.reshape(1, b.shape[0], 1, 1)            # [1,1,4096,1]  -> matches [W,H,C,N] add
            _write_alias(tensors, "glm4v.downsample.bias", b4)

    if hasattr(vis, "merger"):
        pm = vis.merger
        _write_alias(tensors, "glm4v.merger.proj.weight", _as_numpy(pm.proj.weight, dtype))
        if getattr(pm.proj, "bias", None) is not None:
            _write_alias(tensors, "glm4v.merger.proj.bias", _as_numpy(pm.proj.bias, dtype))
        if hasattr(pm, "post_projection_norm"):
            _write_alias(tensors, "glm4v.merger.ln.weight", _as_numpy(pm.post_projection_norm.weight, dtype))
            _write_alias(tensors, "glm4v.merger.ln.bias",   _as_numpy(pm.post_projection_norm.bias, dtype))
        if hasattr(pm, "gate_proj"):
            _write_alias(tensors, "glm4v.merger.mlp.gate", _as_numpy(pm.gate_proj.weight, dtype))
            if getattr(pm.gate_proj, "bias", None) is not None:
                _write_alias(tensors, "glm4v.merger.mlp.gate_bias", _as_numpy(pm.gate_proj.bias, dtype))
        if hasattr(pm, "up_proj"):
            _write_alias(tensors, "glm4v.merger.mlp.up", _as_numpy(pm.up_proj.weight, dtype))
            if getattr(pm.up_proj, "bias", None) is not None:
                _write_alias(tensors, "glm4v.merger.mlp.up_bias", _as_numpy(pm.up_proj.bias, dtype))
        if hasattr(pm, "down_proj"):
            _write_alias(tensors, "glm4v.merger.mlp.down", _as_numpy(pm.down_proj.weight, dtype))
            if getattr(pm.down_proj, "bias", None) is not None:
                _write_alias(tensors, "glm4v.merger.mlp.down_bias", _as_numpy(pm.down_proj.bias, dtype))

    # ---- (5) Flush ----
    for name, arr in tensors:
        if arr is None:
            continue
        if arr.ndim > 4:
            raise RuntimeError(f"Tensor {name} has ndim={arr.ndim} > 4")
        gg.add_tensor(name, arr)

    gg.write_header_to_file()
    gg.write_kv_data_to_file()
    gg.write_tensors_to_file()
    gg.close()

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (e.g., THUDM/GLM-4.5V)")
    ap.add_argument("--outfile", required=True, help="Output .mmproj.gguf path")
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"], help="On‑disk tensor dtype")
    args = ap.parse_args()

    from transformers import Glm4vMoeForConditionalGeneration
    model = Glm4vMoeForConditionalGeneration.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    export(model, args.outfile, dtype=args.dtype)

if __name__ == "__main__":
    main()
