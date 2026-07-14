# PRD: In-browser Needle inference (prd:wasm-needle)

## SOP exception (blanket, user-authorized)

Gates requiring GitHub issue publication (Aligned brief placement, Sliced
per-slice issues, merge-cascade labels, arch-followup issues) are satisfied
locally for this feature: this repo has no git remote by Jason's explicit
choice (grilling Q7, 2026-07-13). Brief lives here; slices are the checklist
below; follow-ups land in `docs/agents/`-adjacent files. Single exception
covers all issue-shaped gates — authorized by the user, highest precedence.

## Goal

Run the finetuned 26M Needle checkpoint entirely in the browser at the
UIModel seam, replacing the server inference path. Motivations (all three,
grilling Q1): kill per-request Python cold-start latency, zero-backend
static deployment, and the demo value of client-side generative UI.

## Decisions (from grilling)

| #   | Decision                                                                                                                                                                                                                                                              |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | Goals: latency + zero-backend + demo — all three                                                                                                                                                                                                                      |
| Q2  | **Replace** the server path: delete `/api/infer` + `scripts/infer-needle.py`                                                                                                                                                                                          |
| Q3  | Acceptance = **accuracy parity**: ≥100% routing / 100% exact args on `finetune/eval.jsonl` (260) and `finetune/regression.jsonl` (8), same harness semantics as `scripts/eval-needle.py`. Constrained-decode port only if unconstrained Python eval shows it's needed |
| Q4  | **Time-boxed ONNX spike (~1h)**: try JAX→ONNX + onnxruntime-web; if it fights back, plan B is exporting raw fp16 weight arrays and hand-writing the ~200-line forward pass in TypeScript                                                                              |
| Q5  | **fp16 weight bundle** (~52MB), lazy-loaded on first command, progress shown in the existing tool-call status line, browser-cached                                                                                                                                    |
| Q6  | **Static export in scope**: `next.config` `output: "export"`; `bun run build` must produce a working static site                                                                                                                                                      |

## Spike results (Q4, recorded post-hoc)

ONNX attempt ran 2026-07-13 inside the time-box: `jax2onnx` failed tracing
the encoder (`nn.scan` dynamic-shape error, `Shapes must be 1D sequences of
concrete values, got JitTracer`). Spike script was scratch-only (not
committed); plan B (fp16 raw weights + hand-written TS forward pass) taken.
The Python-side `--no-constrained` eval scored 100%/100%, so the
constrained-decoding grammar was not ported. Actual forward pass is ~440
lines vs the ~200 estimated — GQA, RoPE, and cross-KV caching are why.

## Modules expected to change

- `lib/weave/model.ts` — new in-browser `UIModel` implementation; fallback chain becomes wasm → mock
- `lib/weave/` (new files) — tokenizer port, decode loop, weight loading
- `app/api/infer/route.ts`, `scripts/infer-needle.py` — **deleted**
- `next.config.*` — static export
- `public/` or equivalent — weight bundle asset
- Export tooling (Python, runs in needle venv) — new script(s) in `scripts/`
- README — seam table + deploy story

## Success criteria (observable)

1. Fresh `bun run build` static export, served by any static file server: typing "Add pick up groceries on Saturday" renders the task with spaces, tool-call line tagged `· wasm` (or equivalent source tag), with no server process.
2. Browser-side parity eval page/harness scores 100% routing + 100% exact args on the 268 eval+regression examples.
3. First command shows load progress; subsequent commands infer without re-downloading (browser cache).
4. `bun run lint` clean, full test suite green.

## Out of scope

- int8 quantization (only if fp16 misses the accuracy bar)
- WebGPU acceleration; CPU WASM/JS is fine at 26M
- Offline/PWA packaging beyond what static export gives for free
- Keeping any server fallback (`/api/infer` is deleted, mock remains as instant-response fallback while weights load)

## Slices (Gate 2 checklist — local stand-in for per-slice issues)

- [x] S1: Export spike — ONNX attempt (time-boxed 1h), else fp16 raw-weight export; plus Python-side `constrained=False` eval to decide if grammar port is needed. Deliverable: weight bundle + a Node-side parity check ≥ acceptance bar
- [x] S2: Browser inference — tokenizer + decode in TS at the UIModel seam, lazy weight load with progress, source tag; parity eval green in-browser
- [x] S3: Replace & static export — delete `/api/infer` + shim, `output: "export"`, README update, end-to-end browser verify of success criterion 1
