#!/usr/bin/env python3
"""PoC: CLM as second-stage re-ranker over BM25 candidates for KB search.

Baseline: search_rank.rank (BM25, field-weighted) over all notes.
CLM: re-rank BM25 top-30 with /v1/rank (query vs candidate title+summary).

Queries are hand-written zh/en with a known target note (matched by title substring).
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "/home/ccc/projects/knowledge_database/99_meta/scripts")
import search_rank  # noqa: E402

KB = Path("/home/ccc/projects/knowledge_database")
CLM = "http://127.0.0.1:8700"
QUESTION = "哪篇笔记最能回答这个查询?"
TOP_K = 30

# (query, substring of target note's title)
QUERIES = [
    ("交叉熵和信息论的关系", "Cross Entropy"),
    ("机器人训练数据供应商有哪些", "Robot Training Data Vendors"),
    ("flow matching 的原理", "Flow Matching"),
    ("动态场景重建 高斯泼溅", "4D Gaussian Splatting"),
    ("因果世界模型 访谈", "Moonlake"),
    ("机器人初创公司怎么招人才", "Sunday"),
    ("DDPM DDIM 训练对比", "DDPM and DDIM"),
    ("SigLIP 怎么改进 CLIP", "SigLIP"),
    ("gripper TCP 选型", "TCP"),
    ("机器人数据质检 人工复核", "数据质检"),
    ("world model 世界模型 机器人", "world"),
    ("diffusion 中的 optimal transport pairing", "OT-Based Pairing"),
]


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    fm, body = (m.group(1), m.group(2)) if m else ("", text)
    f = lambda n: (re.search(rf'^{n}:\s*"(.*?)"\s*$', fm, re.M) or [None, ""])[1]
    t = re.search(r"^tags:\s*\[(.*?)\]\s*$", fm, re.M)
    tags = [x.strip().strip('"').strip("'") for x in t.group(1).split(",")] if t else []
    return {"title": f("title"), "filename": path.stem, "summary": f("summary"),
            "tags": tags, "content": body[:4000]}


def clm_rerank(query: str, docs: list[dict]) -> list[int]:
    texts = [f"{d['title']} — {d['summary'][:200]}" for d in docs]
    payload = json.dumps({"context": query, "question": QUESTION,
                          "answers": texts}).encode()
    req = urllib.request.Request(CLM + "/v1/rank", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        ranked = json.loads(r.read())["ranked"]
    order = []
    seen = {}
    for i, t in enumerate(texts):
        seen.setdefault(t, []).append(i)
    for x in ranked:
        order.append(seen[x["candidate"]].pop(0))
    return order


def main() -> None:
    files = [p for d in ("02_areas", "03_resources") for p in (KB / d).rglob("*.md")]
    docs, paths = [], []
    for p in files:
        try:
            docs.append(parse(p))
            paths.append(p)
        except Exception:
            pass
    print(f"corpus: {len(docs)} notes")

    base_mrr, clm_mrr = [], []
    for query, target_sub in QUERIES:
        target = next((i for i, d in enumerate(docs) if target_sub.lower() in d["title"].lower()), None)
        if target is None:
            print(f"  SKIP (target not found): {target_sub}")
            continue
        base_order = search_rank.rank(query, docs)
        base_rank = base_order.index(target) + 1
        cands = base_order[:TOP_K]
        if target in cands:
            re_order = clm_rerank(query, [docs[i] for i in cands])
            clm_rank_pos = re_order.index(cands.index(target)) + 1
        else:
            clm_rank_pos = None  # never retrieved: re-rank can't help
        base_mrr.append(1.0 / base_rank)
        clm_mrr.append(1.0 / clm_rank_pos if clm_rank_pos else 0.0)
        mark = "" if clm_rank_pos else " (not in BM25 top-30)"
        print(f"  {query[:26]:<28} target《{docs[target]['title'][:22]}》  BM25 #{base_rank:<3} → CLM #{clm_rank_pos}{mark}")

    print("\n=== aggregate ===")
    print(f"queries: {len(base_mrr)}")
    print(f"MRR  BM25 only: {sum(base_mrr)/len(base_mrr):.3f}")
    print(f"MRR  BM25+CLM : {sum(clm_mrr)/len(clm_mrr):.3f}")


if __name__ == "__main__":
    main()
