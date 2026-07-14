"""Dump parity fixtures + measure reference accuracy (constrained=False).

Run from the needle repo context (imports needle). Usage:
    python scripts/dump-fixtures.py <checkpoint.pkl>

Writes fixtures into lib/weave/needle/__fixtures__/ and prints routing/full
accuracy per file (same semantics as scripts/eval-needle.py).
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# needle is importable via its own venv/site; add repo if needed via env.
from needle import SimpleAttentionNetwork, get_tokenizer, load_checkpoint
from needle.model.run import (
    normalize_tools,
    restore_tool_names,
    _build_encoder_input,
)
from needle.model.architecture import make_causal_mask, make_padding_mask
from needle.dataset.dataset import DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_GEN_LEN

BR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(BR, "lib", "weave", "needle", "__fixtures__")
os.makedirs(FIX, exist_ok=True)

EVAL = os.path.join(BR, "finetune", "eval.jsonl")
REG = os.path.join(BR, "finetune", "regression.jsonl")

ENC_PAD = 512


def greedy(model, params, tok, step, query, tools, max_gen_len=DEFAULT_MAX_GEN_LEN,
           max_enc_len=DEFAULT_MAX_ENC_LEN):
    """Mirror generate(..., constrained=False, normalize=True). Returns (text, enc_tokens, gen_ids)."""
    norm_tools, name_map = normalize_tools(tools)
    enc_tokens = _build_encoder_input(tok, query, norm_tools, max_enc_len)
    pad_id, eos_id = tok.pad_token_id, tok.eos_token_id
    real_len = len(enc_tokens)
    padded = enc_tokens + [pad_id] * (ENC_PAD - real_len)
    enc_input = jnp.array([padded])
    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out, enc_mask = model.apply({"params": params}, enc_input,
                                         src_mask=src_mask, method="encode")
    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32).at[0, 0].set(eos_id)

    gen = []
    logits = step(dec_buffer, encoder_out, enc_mask)
    first_step_logits = np.array(logits[0, 0], dtype=np.float32)
    for i in range(max_gen_len - 1):
        nt = int(jnp.argmax(logits[0, i]))
        if nt == eos_id:
            break
        gen.append(nt)
        dec_buffer = dec_buffer.at[0, i + 1].set(nt)
        logits = step(dec_buffer, encoder_out, enc_mask)
    text = tok.decode(gen)
    if text.startswith("<tool_call>"):
        text = text[len("<tool_call>"):]
    if name_map:
        text = restore_tool_names(text, name_map)
    return text, enc_tokens, gen, first_step_logits, np.array(encoder_out[0, :real_len], dtype=np.float32)


def parse_call(text):
    try:
        c = json.loads(text)[0]
        return c["name"], {k: str(v) for k, v in (c.get("arguments") or {}).items()}
    except Exception:
        return None, None


def score(rows, outs):
    route = full = 0
    for r, out in zip(rows, outs):
        want = json.loads(r["answers"])[0]
        want_args = {k: str(v) for k, v in want["arguments"].items()}
        name, args = parse_call(out)
        if name == want["name"]:
            route += 1
            if args == want_args:
                full += 1
    n = len(rows)
    return route, full, n


def main():
    global ENC_PAD
    ckpt = sys.argv[1]
    params, config = load_checkpoint(ckpt)
    model = SimpleAttentionNetwork(config)
    tok = get_tokenizer()

    all_rows = {}
    maxlen = 0
    for path, tag in [(EVAL, "eval"), (REG, "regression")]:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        all_rows[tag] = rows
        for r in rows:
            nt, _ = normalize_tools(r["tools"])
            maxlen = max(maxlen, len(_build_encoder_input(tok, r["query"], nt, DEFAULT_MAX_ENC_LEN)))
    ENC_PAD = maxlen
    print(f"ENC_PAD={ENC_PAD}")

    tgt_mask = make_causal_mask(DEFAULT_MAX_GEN_LEN)

    @jax.jit
    def step(dec, eo, cm):
        return model.apply({"params": params}, dec, eo, self_mask=tgt_mask,
                           cross_mask=cm, method="decode")

    enc_fixtures = []       # {query, tools, enc_tokens}
    e2e_fixtures = []       # {query, tools, gen_ids, text}
    activation = None
    first_logits = []

    for tag in ["eval", "regression"]:
        rows = all_rows[tag]
        outs = []
        for idx, r in enumerate(rows):
            text, enc_tokens, gen, fsl, eo = greedy(model, params, tok, step, r["query"], r["tools"])
            outs.append(text)
            enc_fixtures.append({"query": r["query"], "tools": r["tools"], "enc_tokens": enc_tokens})
            e2e_fixtures.append({"query": r["query"], "tools": r["tools"], "gen_ids": gen, "text": text})
            # verify normalize round-trip identity
            nt, _ = normalize_tools(r["tools"])
            if nt != r["tools"] and idx == 0:
                print(f"WARN normalize changes tools on {tag}[{idx}]")
            if tag == "eval" and idx < 3:
                top5 = np.argsort(-fsl)[:5].tolist()
                first_logits.append({"query": r["query"], "top5": top5,
                                     "top5_vals": [float(fsl[t]) for t in top5]})
                if idx == 0:
                    activation = {"query": r["query"], "tools": r["tools"],
                                  "enc_tokens": enc_tokens,
                                  "encoder_out": eo.tolist(),
                                  "first_step_logits_top5": top5,
                                  "first_step_logits_top5_vals": [float(fsl[t]) for t in top5]}
        route, full, n = score(rows, outs)
        print(f"{tag}: routing {route}/{n} ({route/n:.1%})  full {full}/{n} ({full/n:.1%})")

    # decode fixtures: 20 diverse id->text (from generated sequences + specials)
    decode_cases = []
    seen = set()
    for f in e2e_fixtures:
        key = tuple(f["gen_ids"])
        if f["gen_ids"] and key not in seen:
            decode_cases.append({"ids": f["gen_ids"], "text": tok.decode(f["gen_ids"])})
            seen.add(key)
        if len(decode_cases) >= 20:
            break

    json.dump(enc_fixtures, open(os.path.join(FIX, "encode.json"), "w"))
    json.dump(e2e_fixtures, open(os.path.join(FIX, "e2e.json"), "w"))
    json.dump(decode_cases, open(os.path.join(FIX, "decode.json"), "w"))
    json.dump(first_logits, open(os.path.join(FIX, "first_logits.json"), "w"))
    json.dump(activation, open(os.path.join(FIX, "activation.json"), "w"))
    print(f"wrote fixtures to {FIX}")


if __name__ == "__main__":
    main()
