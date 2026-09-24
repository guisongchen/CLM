#!/usr/bin/env python3
"""CLM tag-suggestion PoC — rank the controlled vocabulary against a note.

Usage:
    suggest_tags.py FILE [--top 10]            # suggest tags for one note
    suggest_tags.py --eval [N] [--seed S]      # evaluate on N sampled notes

The candidate pool is the active registry (domain + topic + format); notes whose
true tags sit outside the pool (candidate/deprecated) are reported separately so
the hit rate is not silently capped.

CLM must be running: systemctl --user start clm  (API on 127.0.0.1:8700)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

KB = Path(__file__).resolve().parents[2]
REGISTRY = KB / "99_meta" / "tag_registry.json"
CLM_URL = "http://127.0.0.1:8700"
QUESTION = "Which tags from the controlled vocabulary best describe this note? Pick the specific topics, the domain(s) it belongs to, and its format."
# Best of 5 phrasing variants tested on 20 held-out notes (recall@7: 57% vs 50% for
# the English slug version): Chinese instruction, slug hyphens rendered as spaces.
NOUL_QUESTION = "这篇笔记是否应打上「{tag}」标签?"
BODY_CHARS = 800


def load_pool() -> tuple[list[str], set[str]]:
    reg = json.loads(REGISTRY.read_text())
    domains = list(reg["domain"])
    topics = [t for ts in reg["topic"].values() for t in ts]
    formats = list(reg["format"])
    pool = domains + topics + formats
    return pool, set(pool)


def parse_note(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    fm, body = (m.group(1), m.group(2)) if m else ("", text)
    def field(name: str) -> str:
        f = re.search(rf'^{name}:\s*"(.*?)"\s*$', fm, re.M)
        return f.group(1) if f else ""
    t = re.search(r"^tags:\s*\[(.*?)\]\s*$", fm, re.M)
    tags = [x.strip().strip('"').strip("'") for x in t.group(1).split(",")] if t else []
    return {"title": field("title"), "summary": field("summary"),
            "tags": [x for x in tags if x], "body": body.strip()}


def state_of(note: dict) -> str:
    return f"Title: {note['title']}\nSummary: {note['summary']}\n\n{note['body'][:BODY_CHARS]}"


def clm_rank(state: str, candidates: list[str]) -> list[dict]:
    payload = json.dumps({"context": state, "question": QUESTION,
                          "answers": candidates}).encode()
    req = urllib.request.Request(CLM_URL + "/v1/rank", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["ranked"]


def clm_noul(state: str, candidates: list[str]) -> list[dict]:
    """Independent p(tag applies) per candidate — multi-label friendly."""
    qs = {t: {"type": "noul", "instructions": NOUL_QUESTION.format(tag=t.replace("-", " "))}
          for t in candidates}
    payload = json.dumps({"state": state, "questions": qs}).encode()
    req = urllib.request.Request(CLM_URL + "/v1/systemone", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    scored = sorted(((a["noul"], qid) for qid, a in out["answers"].items()),
                    reverse=True)
    return [{"candidate": c, "prob": p} for p, c in scored]


def suggest(path: Path, top: int, mode: str = "noul") -> list[dict]:
    pool, _ = load_pool()
    note = parse_note(path)
    fn = clm_noul if mode == "noul" else clm_rank
    return fn(state_of(note), pool)[:top]


def sample_notes(n: int, seed: int) -> list[Path]:
    files = [p for d in ("02_areas", "03_resources")
             for p in (KB / d).rglob("*.md") if not p.name.endswith("_synthesis.md")]
    random.Random(seed).shuffle(files)
    return files[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--eval", type=int, nargs="?", const=20, metavar="N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["rank", "noul"], default="noul")
    args = ap.parse_args()

    if args.eval is None:
        if not args.file:
            ap.error("give a FILE or --eval N")
        for r in suggest(args.file, args.top, args.mode):
            print(f"{r['prob']:.3f}  {r['candidate']}")
        return

    pool, pool_set = load_pool()
    notes = sample_notes(args.eval, args.seed)
    ks = (3, 5, 7, 10)
    hits = {k: 0 for k in ks}
    total_true_in_pool = 0
    out_of_pool = []
    for i, path in enumerate(notes, 1):
        note = parse_note(path)
        true = [t for t in note["tags"] if t in pool_set]
        dropped = [t for t in note["tags"] if t not in pool_set]
        if dropped:
            out_of_pool.append((path.name, dropped))
        if not true:
            continue
        fn = clm_noul if args.mode == "noul" else clm_rank
        ranked = [r["candidate"] for r in fn(state_of(note), pool)]
        total_true_in_pool += len(true)
        for k in ks:
            hits[k] += sum(1 for t in true if t in ranked[:k])
        top5 = ranked[:5]
        mark = lambda t: f"[{t}]" if t in true else t
        print(f"{i:2d}. {note['title'][:38]:<40} true={true}\n"
              f"    top5={[mark(t) for t in top5]}")

    print("\n=== aggregate ===")
    print(f"notes evaluated: {len(notes)}, true tags in pool: {total_true_in_pool}")
    for k in ks:
        print(f"recall@{k:2d}: {hits[k]}/{total_true_in_pool} = {hits[k]/max(1,total_true_in_pool):.1%}")
    if out_of_pool:
        print(f"\ntrue tags outside candidate pool ({len(out_of_pool)} notes):")
        for name, tags in out_of_pool:
            print(f"  {name}: {tags}")


if __name__ == "__main__":
    sys.exit(main())
