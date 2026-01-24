python3 - <<'PY'
import os, json

meta_dir = "ml/feedback/not_a_building/metadata"
ok = 0
total = 0

for fn in os.listdir(meta_dir):
    if not fn.endswith(".json"): 
        continue
    total += 1
    m = json.load(open(os.path.join(meta_dir, fn)))
    aoi = m.get("aoi_bounds")
    if not aoi:
        continue
    if m.get("original_bbox") or m.get("bbox") or m.get("original_geometry"):
        ok += 1

print("total not_a_building:", total)
print("has enough info for bbox:", ok)
print("percent:", (ok/total*100 if total else 0))
PY
