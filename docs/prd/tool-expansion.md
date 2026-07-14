# PRD: Tool catalog expansion (prd:tool-expansion, T2)

## SOP exception

Local PRD instead of GH issue — standing no-remote authorization
(wasm-needle grilling Q7, 2026-07-13).

## Goal

Grow weave's frozen 4-tool catalog to 6 so the app covers the everyday todo
verbs: **edit_task** (rename / reschedule), **uncomplete_task** (undo a
check-off), and a **due filter** on show_tasks ("what's due Saturday").
One generator extension + one GPU finetune cycle + re-export; UI reducers
and suggested commands updated to match.

## Decisions (from grilling)

| #   | Decision                                                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | One tool: `edit_task(title, new_title?, new_due?)` — partial-match `title`, ≥1 optional arg present                                               |
| Q2  | Separate `uncomplete_task(title)` mirroring `complete_task`                                                                                       |
| Q3  | Optional free-string `due` arg on `show_tasks`, same format as add_task's due                                                                     |
| Q4  | UI due matching = case-insensitive substring + small day-name alias table (fri ↔ friday); no date parsing (upgrade path noted in code)            |
| Q5  | Acceptance unchanged: 100%/100% on regenerated eval (6 tools) + extended regression list; Bun parity AND `/dev/parity` green; median ≤700ms holds |
| Q6  | Chat-only UI — reducers + suggested/help command text updated; no new buttons                                                                     |
| Q7  | New checkpoint dir in needle repo (rollback), re-export overwrites `public/needle/weights.bin` in place (gitignored)                              |

## Modules expected to change

- `lib/weave/tools.ts` — 2 new tool schemas + `due` param on show_tasks
- `scripts/gen-finetune-data.ts` — verb templates + arg generators for the
  new tools; regression list extended with new-tool + old-tool cases
- `lib/weave/model.ts` (mock) + reducer in `components/weave-todo.tsx` —
  edit / uncomplete / due-filter handling, suggested-command text
- Needle side (C:\Dev\needle, WSL GPU venv): finetune → eval → export via
  existing scripts; no needle-repo code changes expected
- `scripts/export-weights.py` copies regenerated eval/regression jsonl

## Success criteria (observable)

1. Regenerated eval (6 tools) + extended regression: 100% routing / 100%
   exact args on Bun parity eval and browser `/dev/parity` (WebGPU active).
2. Old 4-tool regression examples still pass (no catastrophic forgetting).
3. In-browser: "rename gym to yoga", "I didn't finish the groceries one",
   "what's due Friday" each produce the right tool call and UI change.
4. Median end-to-end ≤700ms still PASS on `/dev/parity`.
5. lint, tsc, bun test, static export build green.

## Out of scope

- Real date parsing for dues (substring + alias only)
- Per-row edit/uncheck buttons
- Priorities, recurring tasks, tags, clear_completed
- Versioned weight bundles

## Slices

- [x] S1: Catalog + generator — tools.ts schemas, generator templates,
      regenerate train/eval/regression; sanity-check sample rows
- [x] S2: Finetune + export — GPU finetune (batch 16), Python eval 100%/100%,
      export weights + jsonl, Bun parity green
- [x] S3: UI — reducers (edit/uncomplete/due filter + alias table), mock
      model parity, suggested-command text; bun tests
- [x] S4: Verify — browser `/dev/parity` full run, live command checks,
      README touch-up
