# gguf_check_mm.py
from gguf import GGUFReader
import sys
rd = GGUFReader(sys.argv[1])
mm = [t.name for t in rd.tensors if t.name.startswith("mm.")]
print("mm.* tensors:", len(mm))
for n in mm: print(n)