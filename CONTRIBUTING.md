# Contributing

This is a proof-of-concept repo, but it runs a real process. If you want to
poke at it:

## Ground rules

- Engineering principles: KISS/DRY/YAGNI, fail fast and loud, explicit over
  implicit, surgical changes only — match the existing style of whatever
  you touch.
- Feature specs live in `docs/prd/`; architecture followups in
  `docs/arch-followups/`. New behavior gets a PRD note before code.

## The hard invariant

Any change that touches the model, the tool catalog, the data generator, or
the TS/GPU inference paths must keep the parity bar: **100% routing / 100%
exact-args** on `finetune/eval.jsonl` + `finetune/regression.jsonl`, on
every backend (`bun scripts/parity-eval.ts` and `/dev/parity` in Chrome).
The README's "Training + building the weight bundle" section is the full
verification loop.

## Quick checks before any PR

```bash
bun test
bunx tsc --noEmit
bun run lint
bun run build   # static export must succeed
```

## Growing the tool catalog

Adding a tool is a data + retrain job, not just code: schema in
`lib/weave/tools.ts`, ≥100 generated examples per new tool in
`scripts/gen-finetune-data.ts` (plus regression cases), retrain, re-export,
re-verify. Budget ~5 minutes on a CUDA GPU, plus the parity runs.
