#!/usr/bin/env python3
import sys, re
from pathlib import Path

# gguf-py is bundled in llama.cpp; this import matches recent trees
from gguf.gguf_reader import GGUFReader

def collect_kv(reader: GGUFReader) -> dict[str, object]:
    """
    Return {key: value} regardless of gguf-py version.
    Tries several attribute layouts used across commits.
    """
    # 1) Newer trees often have .metadata (plain dict)
    if hasattr(reader, "metadata") and isinstance(reader.metadata, dict):
        return dict(reader.metadata)

    # 2) Some expose .kv (dict of GGUFValue objects)
    if hasattr(reader, "kv") and isinstance(reader.kv, dict):
        return {k: (getattr(v, "value", v)) for k, v in reader.kv.items()}

    # 3) Older: .kv_data (dict of GGUFValue objects)
    if hasattr(reader, "kv_data") and isinstance(reader.kv_data, dict):
        return {k: (getattr(v, "value", v)) for k, v in reader.kv_data.items()}

    # 4) Worst case: try fields (not common; kept as last resort)
    kv = {}
    if hasattr(reader, "fields"):
        for f in reader.fields:
            # Some versions store KVs in .fields with .key/.value
            k = getattr(f, "key", None)
            v = getattr(f, "value", None)
            if k is not None:
                kv[str(k)] = getattr(v, "value", v)
    return kv

def get_kv_num(kv: dict, *keys, default=None, as_int=False):
    for k in keys:
        if k in kv:
            v = kv[k]
            try:
                return int(v) if as_int else float(v)
            except Exception:
                return v
    return default

def main(path: Path):
    rd = GGUFReader(path)
    kv = collect_kv(rd)

    # ---- read vision KVs (use a few common aliases) ----
    n_embd  = get_kv_num(kv,
        "clip.vision.embedding_length", "clip.vision.n_embd", "clip.vision.hidden_size",
        "vision.embedding_length", "vision.n_embd", "vision.hidden_size",
        as_int=True)
    n_ff    = get_kv_num(kv,
        "clip.vision.feed_forward_length", "clip.vision.n_ff", "clip.vision.intermediate_size",
        "vision.feed_forward_length", "vision.n_ff", "vision.intermediate_size",
        as_int=True)
    n_head  = get_kv_num(kv,
        "clip.vision.head_count", "clip.vision.n_head", "clip.vision.num_heads",
        "vision.head_count", "vision.n_head", "vision.num_heads",
        as_int=True)
    n_layer = get_kv_num(kv,
        "clip.vision.block_count", "clip.vision.n_layer", "clip.vision.num_hidden_layers", "clip.vision.depth",
        "vision.block_count",     "vision.n_layer",     "vision.num_hidden_layers",     "vision.depth",
        as_int=True)
    img_sz  = get_kv_num(kv, "clip.vision.image_size", "vision.image_size", as_int=True)
    patch   = get_kv_num(kv, "clip.vision.patch_size", "vision.patch_size", as_int=True)
    projdim = get_kv_num(kv,
        "clip.vision.projection_dim", "clip.image_projection_dim",
        "vision.projection_dim", "image_projection_dim",
        as_int=True)

    print(f"file: {path}")
    print(f"vision: n_embd={n_embd}  n_ff={n_ff}  n_head={n_head}  n_layer={n_layer}  "
          f"image_size={img_sz}  patch={patch}  projection_dim={projdim}")

    # ---- collect tensor shapes ----
    tensors = {t.name: tuple(t.shape) for t in rd.tensors}

    # infer n_layer if missing
    if not n_layer:
        max_blk = -1
        for name in tensors:
            m = re.search(r"^v\.blk\.(\d+)\.", name)
            if m:
                max_blk = max(max_blk, int(m.group(1)))
        n_layer = max_blk + 1 if max_blk >= 0 else None

    # choose blocks to check
    if n_layer is None or n_layer <= 0:
        blocks_to_check = [0, 1, 2]
    else:
        blocks_to_check = sorted({0, 1, max(0, n_layer // 2), n_layer - 1})

    def expect(name, shape):
        got = tensors.get(name)
        ok = (got == shape) if (got and shape) else (got is not None)
        status = "OK" if ok else "MISMATCH"
        print(f"{status:9}  {name:40s}  got={got}  exp={shape}")

    print("\n--- global tensors ---")
    if n_embd and patch:
        expect("v.patch_embd.weight", (n_embd, 3, patch, patch))
    else:
        expect("v.patch_embd.weight", None)  # presence only

    print("\n--- per-block tensors ---")
    for b in blocks_to_check:
        qn = f"v.blk.{b}.attn_q.weight"
        kn = f"v.blk.{b}.attn_k.weight"
        vn = f"v.blk.{b}.attn_v.weight"
        on = f"v.blk.{b}.attn_out.weight"
        ln1w = f"v.blk.{b}.ln1.weight"
        ln2w = f"v.blk.{b}.ln2.weight"
        upn = f"v.blk.{b}.ffn_up.weight"
        dwn = f"v.blk.{b}.ffn_down.weight"

        # Only check shapes we can infer
        expect(qn, (n_embd, n_embd) if n_embd else None)
        expect(kn, (n_embd, n_embd) if n_embd else None)
        expect(vn, (n_embd, n_embd) if n_embd else None)
        expect(on, (n_embd, n_embd) if n_embd else None)
        expect(ln1w, (n_embd,) if n_embd else None)
        expect(ln2w, (n_embd,) if n_embd else None)
        expect(upn, (n_ff, n_embd) if (n_ff and n_embd) else None)
        expect(dwn, (n_embd, n_ff) if (n_ff and n_embd) else None)

    print("\n--- projector tensors (if any) ---")
    for key in ["mm.model.fc.weight", "mm.model.fc.bias",
                "mm.model.mlp.0.weight", "mm.model.mlp.0.bias",
                "mm.model.mlp.2.weight", "mm.model.mlp.2.bias"]:
        if key in tensors:
            print(f"FOUND     {key:40s}  shape={tensors[key]}")
    print("\nDone.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_mmproj.py /path/to/model.mmproj.gguf")
        sys.exit(1)
    main(Path(sys.argv[1]))