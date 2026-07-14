# Arch followups: prd:tool-expansion (queued)

Gate 6 trigger (c): 6 production files touched. Local file per the standing
no-remote exception (labels would be kind:arch-followup, prd:tool-expansion,
queued).

- Per-verb duplication in `lib/weave/api.ts` `callApi` (complete / uncomplete /
  edit / delete all repeat the updateStore + findTask + hit-message shape) and
  in `lib/weave/model.ts` mock regexes. Consolidate only if the catalog grows
  again. severity: low
- Run alongside the still-queued wasm-needle + webgpu-encoder
  `/improve-codebase-architecture` passes.

# Arch followups: prd:command-queue (queued)

Gate 6 trigger (b): new module lib/weave/queue.ts.

- weave now has two FIFO serializers (client queue.ts + worker inline chain);
  consolidate on queue.ts if a third appears. severity: low
