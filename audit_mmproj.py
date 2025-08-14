#!/usr/bin/env python3
import sys, re
from pathlib import Path
from gguf.gguf_reader import GGUFReader

REQUIRED_PATTERNS = [
    r"^v\.patch_embd\.weight$",
    r"^v\.blk\.\d+\.ln1\.weight$",
    r"^v\.blk\.\d+\.attn_q\.weight$",
    r"^v\.blk\.\d+\.attn_k\.weight$",
    r"^v\.blk\.\d+\.attn_v\.weight$",
    r"^v\.blk\.\d+\.attn_out\.weight$",
    r"^v\.blk\.\d+\.ln2\.weight$",
    r"^v\.blk\.\d+\.ffn_up\.weight$",
    r"^v\.blk\.\d+\.ffn_down\.weight$",
]

def main(p):
    rd = GGUFReader(Path(p))
    names = {t.name for t in rd.tensors}
    # infer depth
    depth = -1
    for n in names:
        m = re.search(r"^v\.blk\.(\d+)\.", n)
        if m: depth = max(depth, int(m.group(1)))
    depth = depth + 1 if depth >= 0 else 0
    print(f"depth inferred = {depth}")
    import fnmatch
    import re as _re
    missing = []
    for pat in REQUIRED_PATTERNS:
        rx = _re.compile(pat)
        found = any(rx.match(n) for n in names)
        if not found:
            missing.append(pat)
    if missing:
        print("MISSING classes of tensors (regex):")
        for m in missing: print(" -", m)
    # per-block presence
    per_block_missing = []
    for i in range(depth):
        req = [
            f"v.blk.{i}.ln1.weight",
            f"v.blk.{i}.attn_q.weight",
            f"v.blk.{i}.attn_k.weight",
            f"v.blk.{i}.attn_v.weight",
            f"v.blk.{i}.attn_out.weight",
            f"v.blk.{i}.ln2.weight",
            f"v.blk.{i}.ffn_up.weight",
            f"v.blk.{i}.ffn_down.weight",
        ]
        for r in req:
            if r not in names:
                per_block_missing.append(r)
    if per_block_missing:
        print("Per-block missing tensors:")
        for r in per_block_missing[:50]:  # print first 50
            print(" -", r)
        if len(per_block_missing) > 50:
            print(f"... and {len(per_block_missing)-50} more")
    else:
        print("Per-block required tensors: ALL PRESENT")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audit_required_names.py /path/to/model.mmproj.gguf")
        sys.exit(1)
    main(sys.argv[1])