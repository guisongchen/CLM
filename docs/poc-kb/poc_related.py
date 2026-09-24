#!/usr/bin/env python3
"""PoC: CLM semantic ranking vs shared-tag-count baseline for related-note suggestion.

Ground truth: the agent-curated [[wikilinks]] in each note's `## Related` section.
Candidates: all notes sharing >=1 tag with the source note (same pool kb-agent related uses).
"""
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

KB = Path("/home/ccc/projects/knowledge_database")
CLM = "http://127.0.0.1:8700"
QUESTION = "哪篇笔记与当前笔记内容最相关、最值得互相链接?"


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    fm, body = (m.group(1), m.group(2)) if m else ("", text)
    f = lambda n: (re.search(rf'^{n}:\s*"(.*?)"\s*$', fm, re.M) or [None, ""])[1]
    t = re.search(r"^tags:\s*\[(.*?)\]\s*$", fm, re.M)
    tags = [x.strip().strip('"').strip("'") for x in t.group(1).split(",")] if t else []
    rel = re.search(r"^## Related\s*\n(.*?)(?=^## |\Z)", body, re.S | re.M)
    links = re.findall(r"\[\[([^\]|]+)", rel.group(1)) if rel else []
    return {"stem": path.stem, "title": f("title"), "summary": f("summary"),
            "tags": tags, "links": set(links)}


def clm_rank(state: str, candidates: list[str]) -> list[str]:
    payload = json.dumps({"context": state, "question": QUESTION,
                          "answers": candidates}).encode()
    req = urllib.request.Request(CLM + "/v1/rank", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return [x["candidate"] for x in json.loads(r.read())["ranked"]]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    files = [p for d in ("02_areas", "03_resources") for p in (KB / d).rglob("*.md")]
    notes = {}
    for p in files:
        try:
            notes[p.stem] = parse(p)
        except Exception:
            pass

    tag_index = {}
    for stem, note in notes.items():
        for t in note["tags"]:
            tag_index.setdefault(t, set()).add(stem)

    with_links = [s for s, n_ in notes.items() if n_["links"] & notes.keys()]
    random.Random(42).shuffle(with_links)

    ks = (3, 5)
    base_hits = {k: 0 for k in ks}
    clm_hits = {k: 0 for k in ks}
    total = 0
    evaluated = 0
    for stem in with_links[:n]:
        note = notes[stem]
        truth = note["links"] & notes.keys()
        cand = set()
        for t in note["tags"]:
            cand |= tag_index.get(t, set())
        cand -= {stem}
        if not cand or not truth:
            continue
        evaluated += 1
        total += len(truth)
        # baseline: more shared tags first
        baseline = sorted(cand, key=lambda c: -len(set(note["tags"]) & set(notes[c]["tags"])))
        # CLM: semantic rank over candidate title+summary
        state = f"Title: {note['title']}\nSummary: {note['summary']}"
        cand_texts = {c: f"{notes[c]['title']} — {notes[c]['summary'][:150]}" for c in cand}
        ranked_texts = clm_rank(state, list(cand_texts.values()))
        text2stem = {v: k for k, v in cand_texts.items()}
        clm_order = [text2stem[t] for t in ranked_texts]
        for k in ks:
            base_hits[k] += sum(1 for t in truth if t in baseline[:k])
            clm_hits[k] += sum(1 for t in truth if t in clm_order[:k])
        print(f"{note['title'][:34]:<36} cands={len(cand):3d} truth={len(truth)} "
              f"base@5={sum(1 for t in truth if t in baseline[:5])} clm@5={sum(1 for t in truth if t in clm_order[:5])}")

    print("\n=== aggregate ===")
    print(f"notes: {evaluated}, curated links: {total}")
    for k in ks:
        print(f"recall@{k}: baseline {base_hits[k]/total:.1%}  vs  CLM {clm_hits[k]/total:.1%}")


if __name__ == "__main__":
    main()
