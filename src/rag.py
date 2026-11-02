"""Answer questions: fuse retrieval, build a cited prompt, generate with Qwen; optional no-RAG baseline."""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import DenseRetriever
from reranker import Reranker, fuse

import os
LLM = os.environ.get("MEDRAG_LLM", "Qwen/Qwen2.5-7B-Instruct")
SYS_RAG = ("You answer questions about FDA drug labels. Use ONLY the numbered context passages. "
           "Cite every claim with its passage number, like [1]. If the passages do not contain "
           "the answer, reply exactly: I could not find this in the provided labels. "
           "State only what the labels say; do not add medical advice. If a dose or limit differs by patient group (adult, pediatric, renal impairment), state which group each value applies to.")
SYS_PLAIN = "You answer questions about FDA drug labels from your own knowledge. Be concise."

class RAG:
    def __init__(self, size=256, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.ret = DenseRetriever(size=size, device=device)
        self.rr = Reranker(device=device)
        self.tok = AutoTokenizer.from_pretrained(LLM)
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.llm = AutoModelForCausalLM.from_pretrained(LLM, dtype=dt, device_map="auto")

    def _chat(self, messages, max_new_tokens=400):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.llm.device)
        out = self.llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def answer(self, q, k=5, use_rag=True):
        if not use_rag:
            return self._chat([{"role": "system", "content": SYS_PLAIN},
                               {"role": "user", "content": q}]), []
        top = fuse(q, self.ret.search(q, 20), self.rr, k)
        ctx = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(top))
        msgs = [{"role": "system", "content": SYS_RAG},
                {"role": "user", "content": f"Context passages:\n{ctx}\n\nQuestion: {q}"}]
        return self._chat(msgs), top

if __name__ == "__main__":
    rag = RAG(size=256)
    qs = sys.argv[1:] or ["Can sertraline be taken during pregnancy?",
                          "Does warfarin interact with NSAIDs like ibuprofen?",
                          "What is the maximum daily dose of metformin?",
                          "What is the capital of France?"]
    for q in qs:
        for use_rag in (True, False):
            a, src = rag.answer(q, use_rag=use_rag)
            print(f"\nQ: {q}\n[{'RAG' if use_rag else 'NO-RAG'}] {a}")
            for i, h in enumerate(src):
                print(f"   [{i+1}] {h['drug']} | {h['section']}")
