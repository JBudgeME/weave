"""Eval a Needle checkpoint on weave finetune JSONL splits.

Usage (from the needle venv):
    python scripts/eval-needle.py <checkpoint.pkl> <split.jsonl> [more.jsonl ...]

Reports routing accuracy (tool name) and full accuracy (tool + exact args).
"""
import json
import sys

from needle import (
    SimpleAttentionNetwork,
    generate_batch,
    get_tokenizer,
    load_checkpoint,
)


def parse_call(text):
    try:
        calls = json.loads(text)
        c = calls[0]
        return c["name"], {k: str(v) for k, v in (c.get("arguments") or {}).items()}
    except Exception:
        return None, None


def main():
    args = [a for a in sys.argv[1:] if a != "--no-constrained"]
    constrained = "--no-constrained" not in sys.argv
    ckpt, *files = args
    params, config = load_checkpoint(ckpt)
    model = SimpleAttentionNetwork(config)
    tok = get_tokenizer()

    for path in files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        outs = []
        # ponytail: batch of 32 to bound memory; one giant batch OOMs on CPU jax.
        for i in range(0, len(rows), 32):
            chunk = rows[i : i + 32]
            outs += generate_batch(
                model, params, tok,
                [r["query"] for r in chunk],
                [r["tools"] for r in chunk],
                constrained=constrained,
            )
        route_ok = full_ok = 0
        misses = []
        for r, out in zip(rows, outs):
            want = json.loads(r["answers"])[0]
            want_args = {k: str(v) for k, v in want["arguments"].items()}
            name, args = parse_call(out)
            if name == want["name"]:
                route_ok += 1
                if args == want_args:
                    full_ok += 1
                else:
                    misses.append((r["query"], want_args, args))
            else:
                misses.append((r["query"], want["name"], name))
        n = len(rows)
        print(f"{path}: routing {route_ok}/{n} ({route_ok/n:.1%})  "
              f"full {full_ok}/{n} ({full_ok/n:.1%})")
        for q, want, got in misses[:10]:
            print(f"  MISS {q!r}: want {want} got {got}")


if __name__ == "__main__":
    main()
