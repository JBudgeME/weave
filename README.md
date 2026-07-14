# weave 🧵

**The most over-engineered todo app ever built** — a proof of concept for
tiny-model generative UI, running a finetuned 26M-parameter transformer
entirely in your browser.

## TL;DR

You type plain English ("Add pick up groceries on Saturday", "What's due
Friday?") into a todo app. A small neural network — downloaded once,
running locally in your browser tab, no server, no API keys, nothing leaves
your machine — reads the sentence and picks the right action. On a machine
with WebGPU it answers in about a third of a second. The whole "backend" is
a 52MB weight file and some TypeScript.

The point isn't the todo list. The point is: **a 26M-parameter model,
finetuned on a few thousand generated examples in ~3 minutes on a consumer
GPU, can do reliable natural-language → tool-call routing 100% of the time,
client-side.** You don't need a frontier model — or a server — for a narrow,
well-defined command surface.

## What this is

Natural language goes into a command bar. A 26M-parameter model
([Needle](https://github.com/cactus-compute/needle), distilled from Gemini)
emits a single **tool call** — never free-form JSON structure. The task card
re-renders through [json-render](https://github.com/vercel-labs/json-render)
from a hand-written template, so UI output is valid by construction. The
model only ever picks a tool and fills its args:

```
command bar → UIModel.infer() → ToolCall → task store → template → json-render Spec → <Renderer/>
```

Commands queue FIFO client-side, so the input never locks while the model
thinks. Every answer is tagged with the backend that produced it
(`· webgpu` / `· wasm` / `· mock`).

### Tool catalog (frozen per finetune cycle)

| Tool              | Args                                |
| ----------------- | ----------------------------------- |
| `show_tasks`      | `filter`: all / open / done, `due?` |
| `add_task`        | `title`, `due?`                     |
| `complete_task`   | `title` (partial match)             |
| `edit_task`       | `title`, `new_title?`, `new_due?`   |
| `uncomplete_task` | `title` (partial match)             |
| `delete_task`     | `title` (partial match)             |

Growing the catalog means regenerating training data and re-finetuning —
about a 5-minute cycle end to end on a consumer GPU.

## What we were exploring

1. **Can a tiny model be trusted with UI?** Yes — if the model only picks
   from a frozen tool catalog and templates render the result, the UI is
   valid by construction. The model can be wrong about _which_ tool, never
   about _what HTML_.
2. **How small is "good enough"?** Zero-shot, Needle scored 81.5% routing /
   56.2% exact-args on our held-out set. After one finetune on ~3,900
   generated examples: **100% / 100%** — held across three retrains and a
   50% catalog expansion.
3. **How fast can in-browser inference get?** The latency ladder we walked:
   ~15s (server-side CLI, cold) → 5.6s (pure-TS port in a Web Worker) →
   2.8s (KV-cached decoder, 9.4× decode speedup) → **~350–430ms median**
   (encoder + cross-attention projection in WebGPU compute shaders, decoder
   in TS). All three backends produce bit-identical greedy output.
4. **Do you need constrained decoding?** Zero-shot, yes-ish. Post-finetune,
   no: unconstrained greedy decode scores identically, so the browser port
   skips grammar machinery entirely.

## How it was built (paraphrased history)

- **Data**: a deterministic, seeded slot-fill generator
  (`scripts/gen-finetune-data.ts`) emits ~4,300 examples over 40 multi-word
  titles × verb templates × due-date formats, split train/eval, plus a
  hand-picked regression set of documented live misses that is _never_
  trained on.
- **Finetune**: `needle finetune` (JAX) — ~3 minutes on an RTX 4070 Ti
  Super under WSL2. Batch size 16 avoids OOM on 16GB cards.
- **Port**: the entire forward pass (encoder-decoder, GQA attention,
  half-split RoPE, ZCRMSNorm, sigmoid-gated residuals, no FFN) was
  hand-ported to dependency-free TypeScript (`lib/weave/needle/`), computing
  in fp32 over an fp16 weight bundle. An ONNX export was attempted first and
  abandoned (JAX `nn.scan` traced dynamic shapes).
- **Verify**: a parity harness treats the Python reference as the oracle —
  every backend (Bun CLI, browser TS, browser WebGPU) must reproduce 100%
  routing / 100% exact args on the eval + regression splits. The browser
  harness lives at `/dev/parity` with per-phase timings and asserted
  PASS/FAIL verdicts.
- **Speed**: profiling showed decode dominated → KV cache (`DecodeSession`);
  then the encoder + the one-shot cross-attention K/V projection dominated →
  WGSL compute shaders (f32), with GPU-vs-TS max abs diff ≈ 5e-6.
- **Resilience**: inference runs in a Web Worker (UI never blocks). GPU
  failures (device lost, timeouts) demote to the TS path — never to the
  regex mock, which only answers if the weight bundle can't load at all.

Fun war story: the original "model squashes spaces" bug ("pickupgroceries")
was never the model — the upstream CLI streams token-by-token and drops
SentencePiece word-boundary spaces. The model was fine all along.

## Requirements

**To run the app (inference only):**

- [Bun](https://bun.sh) ≥ 1.x
- A browser. WebGPU (Chrome/Edge 113+) gets ~350–430ms commands; anything
  else falls back to the pure-TS path (~2.8s); no JS weight-bundle load at
  all falls back to a regex mock.
- The weight bundle (`public/needle/` — weights.bin ~52MB, manifest,
  tokenizer, eval JSONLs). **It is gitignored**: build it yourself via the
  training steps below.

**To train / regenerate the bundle:**

- Python 3.10+ with [Needle](https://github.com/cactus-compute/needle)
  installed (JAX). CPU works (~2–3 h per finetune); a CUDA GPU does it in
  ~3 min. On Windows, JAX has no native CUDA — use WSL2 Ubuntu with the CUDA
  driver, and create the venv there.
- ~16GB GPU VRAM is plenty with `--batch-size 16`.

## Installation

```bash
git clone https://github.com/JBudgeME/weave.git
cd weave
bun install
bun dev          # http://localhost:3000 — regex-mock mode until you build the bundle
```

## Training + building the weight bundle

```bash
# 1. Generate data (deterministic, seeded — reproducible byte-for-byte)
bun scripts/gen-finetune-data.ts          # → finetune/{train,eval,regression}.jsonl

# 2. Finetune (needle venv; GPU strongly recommended)
needle finetune finetune/train.jsonl --batch-size 16

# 3. Eval the checkpoint — demand 100%/100% before shipping
python scripts/eval-needle.py <ckpt.pkl> finetune/eval.jsonl finetune/regression.jsonl

# 4. Export the browser bundle (weights + tokenizer + manifest + eval data)
python scripts/export-weights.py <ckpt.pkl>

# 5. Regenerate the numeric parity fixtures for the test suite
python scripts/dump-fixtures.py <ckpt.pkl>

# 6. Verify the TS port against the new checkpoint
bun scripts/parity-eval.ts                # must print 100%/100% on both splits
bun test
```

Then open `/dev/parity` in Chrome and hit **run** for the browser-side
verdict (accuracy / GPU-vs-TS diff / median latency, each asserted
PASS/FAIL). Keep that tab focused — background tabs throttle timers and
slow the harness (not the model).

## Deploying

Static export: `bun run build` → deploy `out/` to any static host (GitHub
Pages, Cloudflare Pages, nginx, `bunx serve out`). No server, no env vars.
The only artifact beyond HTML/JS is the `needle/` bundle dir.

## Architecture notes for the curious

- **Swap points**: `lib/weave/wasm-model.ts` (the `UIModel` seam — any
  query→ToolCall backend plugs in here), `lib/weave/store.ts` (localStorage
  → real backend), `lib/weave/templates.ts` (task card template).
- **Model internals** (`lib/weave/needle/`): d=512, 12 encoder / 8 decoder
  layers, GQA with 4 KV heads, no feed-forward blocks, SentencePiece BPE
  vocab 8192. `model.ts` is the fp32 forward pass; `gpu/` holds the WGSL
  shaders + encoder dispatch; `infer.ts` orchestrates greedy decode with a
  per-generate KV-cache `DecodeSession`.
- **Worker protocol** (`lib/weave/needle-worker.ts`): load/progress/ready/
  infer/result/selfcheck/error, FIFO-serialized; `wasm-model.ts` is the
  main-thread client with crash recovery.
- **Perf on the dev machine** (Chrome, RTX 4070 Ti Super): median ~350–430ms
  end-to-end per command (encode ~40–70ms GPU, decode ~300–360ms TS).
- **Known ceilings, by design**: due dates are free strings matched by
  substring + day-name aliases (no date parsing); one tool call per
  utterance (compound commands would need multi-call training data);
  decoder still on CPU (the next obvious speed increment).

## Repo tour

```
components/weave-todo.tsx   UI: command bar, FIFO queue, task card
lib/weave/                  tools catalog, api/store/templates, mock model
lib/weave/needle/           TS port of the model (+ __fixtures__ parity data)
lib/weave/needle/gpu/       WebGPU device/shaders/encoder
lib/weave/needle-worker.ts  Web Worker host
app/dev/parity/             browser parity + latency harness
scripts/                    data generator, eval, export, fixtures, parity
finetune/                   generated JSONL (gitignored)
docs/prd/                   feature PRDs (the process artifacts are real)
```

Scaffolded from a greenfield template (Next.js 16, React 19, Tailwind v4,
shadcn/ui, bun).

## Licensing & provenance

Code is [MIT](LICENSE). Model weights are **not** distributed with this repo
(gitignored); if you train your own checkpoint you inherit the terms of
[Needle](https://github.com/cactus-compute/needle) and its base model
(distilled from Gemini) for that artifact. Contributions: see
[CONTRIBUTING.md](CONTRIBUTING.md).
