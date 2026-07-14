# Issue: client-side command queue (prd:command-queue, T1)

## SOP exception

Local issue file instead of GH issue — standing no-remote authorization
(wasm-needle grilling Q7, 2026-07-13).

## Brief (Gate 1, approved in conversation 2026-07-14)

Typing shouldn't wait on inference. Keep the command input live while a
command runs; queue submissions client-side and execute them FIFO. The
worker already serializes requests — this only unlocks the textbox.

- In scope: `components/weave-todo.tsx` only. Input + Send stay enabled
  while busy; submitted commands run strictly in order; each result still
  shows its tool-call line; suggestion chips queue too.
- Out of scope: compound utterances → multiple tool calls (needs retrain);
  queue display UI beyond what exists; cancellation.
- Success: type two commands back-to-back — both execute in order, UI never
  blocks; tests cover FIFO ordering; suite/lint/tsc green.

## Slices

- [x] S1: FIFO run queue in weave-todo, input never disabled, tests
