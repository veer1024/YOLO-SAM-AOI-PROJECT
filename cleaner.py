import os, json

BASE = "ml/feedback/missing_building"
BAD_IDS = [
    "3c6c9a58-4c49-4e45-a02b-dcbb6b8df623",
    "5c28218a-1838-444d-8075-c6cb024fba42",
    "6e10717d-17d2-46a8-9982-a5e281ad3fd5",
    "23f21066-c39a-477f-b0b4-cf26e79e393c",
    "54b2abe2-5cd3-4bf3-8e49-69c14dfcc6e5",
    "68b0c428-483d-4013-b418-1f7aae05bfbc",
    "439fa873-94d1-4231-9d59-e200b747c4da",
    "545e75e3-a055-4686-b0e8-afd9e97fe330",
    "b3a9e66b-cf7e-4196-b9ec-0fb68ddcac6b",
    "60af1cee-bddd-4a0b-891e-2f783dc0f3dd",
    "54c9c790-6bd5-4a4a-af97-33ebdec9726b"
]

# Remove files
for fid in BAD_IDS:
    for sub in ["images", "masks", "metadata"]:
        path = f"{BASE}/{sub}/{fid}.png" if sub != "metadata" else f"{BASE}/{sub}/{fid}.json"
        if os.path.exists(path):
            os.remove(path)
            print("deleted", path)

# Clean index.json
index_path = "ml/feedback/index.json"
if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)

    index = [e for e in index if e["id"] not in BAD_IDS]

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print("index.json cleaned")
