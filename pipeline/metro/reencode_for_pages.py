"""Re-encode the DZI pyramid at lower WebP quality to fit GitHub Pages (1 GB)."""
import os, sys
from multiprocessing import Pool
from PIL import Image
SRC, DST, Q = sys.argv[1], sys.argv[2], int(sys.argv[3])
def job(rel):
    out = os.path.join(DST, rel)
    if os.path.exists(out):
        return os.path.getsize(out)
    Image.open(os.path.join(SRC, rel)).convert("RGB").save(out, "WEBP", quality=Q, method=4)
    return os.path.getsize(out)
rels = []
for lvl in os.listdir(SRC):
    os.makedirs(os.path.join(DST, lvl), exist_ok=True)
    rels += [os.path.join(lvl, f) for f in os.listdir(os.path.join(SRC, lvl))]
with Pool(4) as p:
    total = sum(p.imap_unordered(job, rels, chunksize=64))
print(f"{len(rels)} tiles, {total/1e9:.3f} GB at q{Q}")
