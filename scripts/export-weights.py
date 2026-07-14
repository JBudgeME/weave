"""Export a Needle checkpoint to a runtime-agnostic fp16 weight blob.

Usage (needle venv, from anywhere):
    python scripts/export-weights.py <checkpoint.pkl>

Writes into browser-render/public/needle/ (gitignored — regenerate via this
script; weights are ~50MB):
  weights.bin    fp16 little-endian, tensors concatenated in sorted-name order
  manifest.json  { config, rope, special, tensors: {name: {offset, shape, dtype}} }
  tokenizer.json { pieces, scores, types, ...special ids + normalizer flags }

Weights are stored as the raw fp16 checkpoint values. TS decodes fp16 -> fp32
and computes in fp32 (the reference casts fp16 -> bf16; fp16 is strictly more
precise, so this is a faithful upper bound).
"""
import json
import os
import shutil
import sys

import numpy as np

from needle import load_checkpoint, get_tokenizer
from needle.dataset.tokenizer import (
    PAD_ID, EOS_ID, BOS_ID, UNK_ID, TOOL_CALL_ID, TOOLS_ID, TOKENIZER_PREFIX,
)
from needle.dataset.dataset import DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_GEN_LEN

import jax

BR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BR, "public", "needle")
os.makedirs(OUT, exist_ok=True)


def flatten(params):
    """Return sorted list of (dotted_name, np.ndarray)."""
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    out = []
    for kp, v in leaves:
        name = ".".join(str(k.key) if hasattr(k, "key") else str(k) for k in kp)
        out.append((name, np.asarray(v, dtype=np.float32)))
    out.sort(key=lambda x: x[0])
    return out


def main():
    ckpt = sys.argv[1]
    params, config = load_checkpoint(ckpt)
    tensors = flatten(params)

    manifest_tensors = {}
    blob = bytearray()
    offset = 0
    for name, arr in tensors:
        fp16 = arr.astype("<f2")  # little-endian float16
        b = fp16.tobytes()
        manifest_tensors[name] = {
            "offset": offset,
            "shape": list(arr.shape),
            "dtype": "float16",
        }
        blob += b
        offset += len(b)

    head_dim = config.d_model // config.num_heads
    manifest = {
        "config": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "num_encoder_layers": config.num_encoder_layers,
            "num_decoder_layers": config.num_decoder_layers,
            "d_ff": config.d_ff,
            "head_dim": head_dim,
            "rope_theta": config.rope_theta,
            "activation": config.activation,
            "no_feedforward": config.no_feedforward,
            "pad_token_id": config.pad_token_id,
            "eps": 1e-6,
            "max_enc_len": DEFAULT_MAX_ENC_LEN,
            "max_gen_len": DEFAULT_MAX_GEN_LEN,
        },
        "rope": {"theta": config.rope_theta, "head_dim": head_dim},
        "special": {
            "pad": PAD_ID, "eos": EOS_ID, "bos": BOS_ID, "unk": UNK_ID,
            "tool_call": TOOL_CALL_ID, "tools": TOOLS_ID,
        },
        "total_bytes": offset,
        "tensors": manifest_tensors,
    }

    with open(os.path.join(OUT, "weights.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    # ---- tokenizer.json ----
    from sentencepiece import sentencepiece_model_pb2 as spm_pb2
    proto = spm_pb2.ModelProto()
    proto.ParseFromString(open(TOKENIZER_PREFIX + ".model", "rb").read())
    n = proto.normalizer_spec
    pieces = [p.piece for p in proto.pieces]
    scores = [p.score for p in proto.pieces]
    types = [p.type for p in proto.pieces]
    tok = {
        "pieces": pieces,
        "scores": scores,
        "types": types,  # 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 6 BYTE
        "unk_id": UNK_ID,
        "special": manifest["special"],
        "normalizer": {
            "add_dummy_prefix": n.add_dummy_prefix,
            "remove_extra_whitespaces": n.remove_extra_whitespaces,
            "escape_whitespaces": n.escape_whitespaces,
        },
        "byte_fallback": bool(proto.trainer_spec.byte_fallback),
    }
    with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
        json.dump(tok, f)

    # Keep the browser parity data in lockstep with the exported weights —
    # a stale eval set silently tests the new model against old answers.
    for split in ("eval.jsonl", "regression.jsonl"):
        src = os.path.join("finetune", split)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, split))
            print(f"copied {split}")

    print(f"weights.bin: {offset/1e6:.1f} MB, {len(tensors)} tensors")
    print(f"tokenizer.json: {len(pieces)} pieces")
    print(f"wrote to {OUT}")


if __name__ == "__main__":
    main()
