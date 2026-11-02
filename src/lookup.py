import json, sys
drug, section = sys.argv[1].lower(), sys.argv[2]
for l in open("data/processed/chunks_256.jsonl"):
    c = json.loads(l)
    if drug in c["drug"] and c["section"] == section:
        print(f"\n--- {c['drug']} | {c['section']} | chunk {c['chunk_idx']} ---")
        print(c["text"].split("\n", 1)[1][:700])
