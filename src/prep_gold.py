"""Write the source text for each eval question to a helper file for writing gold answers."""
import json
chunks = [json.loads(l) for l in open("data/processed/chunks_256.jsonl")]
out = []
for i, l in enumerate(open("eval/qa_template.jsonl"), 1):
    x = json.loads(l)
    if x["answer"] == "NOT_FOUND":
        continue
    hits = [c for c in chunks if x["drug"] in c["drug"] and c["section"] == x["section"]]
    out.append(f"\n##### Q{i}: {x['q']}  [{x['drug']} | {x['section']}]")
    if not hits:
        out.append("!!! NO CHUNKS FOUND: change the section or drop this question")
    for c in hits[:3]:
        out.append(c["text"].split("\n", 1)[1][:600])
open("eval/gold_helper.txt", "w").write("\n".join(out))
print("wrote eval/gold_helper.txt")
