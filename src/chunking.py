import re, json, argparse
from pathlib import Path
from transformers import AutoTokenizer

SECTIONS = {
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "overdosage": "Overdosage",
    "mechanism_of_action": "Mechanism of Action",
    "description": "Description",
}

SECTIONS.update({
    "boxed_warning": "Boxed Warning",
    "clinical_pharmacology": "Clinical Pharmacology",
    "pharmacokinetics": "Pharmacokinetics",
    "pharmacodynamics": "Pharmacodynamics",
    "information_for_patients": "Patient Information",
    "drug_abuse_and_dependence": "Drug Abuse and Dependence",
    "laboratory_tests": "Laboratory Tests",
    "how_supplied": "How Supplied",
    "nursing_mothers": "Nursing Mothers",
    "pediatric_use": "Pediatric Use",
    "geriatric_use": "Geriatric Use",
    "general_precautions": "General Precautions",
    "spl_medguide": "Medication Guide",
})

tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
tok.model_max_length = 10**6   # we only count tokens here, so silence the warning

def n_tokens(text):
    return len(tok.encode(text, add_special_tokens=False))

def alpha_ratio(s):
    ns = [c for c in s if not c.isspace()]
    return sum(c.isalpha() for c in ns) / max(1, len(ns))

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def split_long(sentence, max_tokens):
    """Hard-split an over-long sentence by tokens (tables, run-ons)."""
    ids = tok.encode(sentence, add_special_tokens=False)
    step = int(max_tokens * 0.8)
    return [tok.decode(ids[i:i + step]) for i in range(0, len(ids), step)]

def chunk_text(text, max_tokens, overlap_tokens):
    sentences = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        sentences += split_long(s, max_tokens) if n_tokens(s) > max_tokens else [s]

    chunks, cur = [], []                      # cur = list of (sentence, n_tokens)
    for s in sentences:
        l = n_tokens(s)
        if cur and sum(x[1] for x in cur) + l > max_tokens:
            chunks.append(" ".join(x[0] for x in cur))
            keep, kl = [], 0                  # carry last sentences as overlap
            for x in reversed(cur):
                if kl + x[1] > overlap_tokens:
                    break
                keep.insert(0, x); kl += x[1]
            cur = keep
        cur.append((s, l))
    if cur:
        chunks.append(" ".join(x[0] for x in cur))
    return chunks

def main(size, overlap):
    out, records = Path(f"data/processed/chunks_{size}.jsonl"), []
    for fp in sorted(Path("data/raw").glob("*.json")):
        d = json.loads(fp.read_text())
        generic = d.get("openfda", {}).get("generic_name", [fp.stem])[0]
        brand = d.get("openfda", {}).get("brand_name", [""])[0]
        for key, section in SECTIONS.items():
            if key not in d:
                continue
            text = clean(" ".join(d[key]))
            if len(text) < 50:
                continue
            # reserve ~30 tokens for the "Drug | Section" header
            for i, c in enumerate(chunk_text(text, size - 60, overlap)):
                if alpha_ratio(c) < 0.6:   # drop table/number noise
                    continue
                records.append({
                    "id": f"{fp.stem}|{key}|{i}",
                    "drug": generic.lower(), "brand": brand,
                    "section": section, "chunk_idx": i,
                    "text": f"Drug: {generic} | Section: {section}\n{c}",
                })
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    lens = [n_tokens(r["text"]) for r in records]
    print(f"size={size} overlap={overlap}: {len(records)} chunks, "
          f"avg {sum(lens)/len(lens):.0f} tokens, max {max(lens)}")
    print("sample:\n", records[len(records)//2]["text"][:600])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=40)
    a = ap.parse_args()
    main(a.size, a.overlap)
