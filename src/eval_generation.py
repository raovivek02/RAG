"""Run RAG and no-RAG on eval/qa.jsonl; report accuracy, false refusals and correct refusals."""
import json, re, sys
from rag import RAG

size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
qa = [json.loads(l) for l in open("eval/qa.jsonl") if l.strip()]
norm = lambda s: re.sub(r"[\s,]+", "", s.lower())
rag = RAG(size=size)

out = []
for x in qa:
    for use_rag in (True, False):
        a, src = rag.answer(x["q"], use_rag=use_rag)
        out.append({"q": x["q"], "gold": x["answer"], "mode": "RAG" if use_rag else "NO-RAG",
                    "answer": a, "sources": [f"{h['drug']}|{h['section']}" for h in src]})
json.dump(out, open("results/generation.json", "w"), indent=1)

ans = [x for x in qa if x["answer"] != "NOT_FOUND"]
for mode in ("RAG", "NO-RAG"):
    rs = {o["q"]: o["answer"] for o in out if o["mode"] == mode}
    acc = sum(norm(x["answer"]) in norm(rs[x["q"]]) for x in ans) / max(1, len(ans))
    print(f"{mode:<7} answer accuracy on answerable questions: {acc:.2f}")
    if mode == "RAG":
        refuse = lambda a: "could not find" in a.lower()
        nf = [x for x in qa if x["answer"] == "NOT_FOUND"]
        print(f"RAG     false refusals on answerable: {sum(refuse(rs[x['q']]) for x in ans)}/{len(ans)}")
        print(f"RAG     correct refusals on unanswerable: {sum(refuse(rs[x['q']]) for x in nf)}/{len(nf)}")
