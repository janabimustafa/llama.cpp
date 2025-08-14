# list_projector_keys.py
import sys, safetensors.torch as st
from pathlib import Path

pt = {}
for f in Path(sys.argv[1]).glob("*.safetensors"):
    pt.update(st.load_file(f, device="cpu"))

cands = [k for k in pt.keys() if any(x in k for x in (
    "mm_projector", "mmprojector", "projector", "vision_proj", "image_proj",
    "modal_project", "multimodal", "multi_modal", "proj."
))]
# ignore ones that start with model.language_model
#cands = [k for k in cands if not k.startswith("model.language_model")]
#combine layers (model.language_model.layers.<layer_id>...) into model.language_model.layers.n...



for k in sorted(cands):
    print(k, pt[k].shape)