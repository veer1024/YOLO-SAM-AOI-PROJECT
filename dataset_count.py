from collections import Counter
import json

DATASET = "ml/feedback/feedback_dataset_clean.json"

with open(DATASET) as f:
    data = json.load(f)

type_counts = Counter(d.get("type", "unknown") for d in data)

print("\n📊 Feedback sample counts:")
for k, v in type_counts.items():
    print(f"  {k:16s}: {v}")

print(f"\nTotal samples: {len(data)}")
