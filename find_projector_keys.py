# find_projector_keys.py
from pathlib import Path
import safetensors.torch as st
import sys
pt = {}
for f in Path(sys.argv[1]).glob("*.safetensors"): pt.update(st.load_file(f, device="cpu"))
for k in sorted(pt):
    if any(x in k for x in ("mm_projector","vision_projector","projector","vision_proj","image_proj","modal_project")):
        print(k, pt[k].shape)