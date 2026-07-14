"use client";

import { useEffect, useRef, useState } from "react";
import type { Spec } from "@json-render/core";
import { JSONUIProvider, Renderer } from "@json-render/react";
import { toast } from "sonner";

import { callApi, type TaskData } from "@/lib/weave/api";
import { type ToolCall } from "@/lib/weave/model";
import { createQueue } from "@/lib/weave/queue";
import { loadNeedle, wasmNeedle } from "@/lib/weave/wasm-model";
import { registry } from "@/lib/weave/registry";
import { tasksSpec } from "@/lib/weave/templates";

const SUGGESTIONS = [
  "Add buy milk",
  "Mark buy milk as done",
  "Rename practice guitar to practice piano",
  "I didn't finish buy milk",
  "What's due Friday?",
  "Delete buy milk",
];

function toolCallLabel(call: ToolCall): string {
  const args = Object.entries(call.args)
    .map(([k, v]) => `${k}: "${v}"`)
    .join(", ");
  return `${call.tool}(${args})${call.source ? ` · ${call.source}` : ""}`;
}

export function WeaveTodo() {
  const [spec, setSpec] = useState<Spec | null>(null);
  // Ref, not state: queued execs must see the *latest* filter, not the one
  // captured when the command was typed (review: stale-closure bug).
  const filterRef = useRef("all");
  const [query, setQuery] = useState("");
  const [pending, setPending] = useState(0);
  const [lastCall, setLastCall] = useState<ToolCall | null>(null);
  const [loadPct, setLoadPct] = useState<number | null>(null);

  function fetchTasks(nextFilter: string): Promise<TaskData> {
    return callApi({
      tool: "show_tasks",
      args: { filter: nextFilter },
    }) as Promise<TaskData>;
  }

  useEffect(() => {
    void fetchTasks("all").then((d) => setSpec(tasksSpec(d)));
  }, []);

  // Commands queue FIFO so the input never locks while one is running; the
  // worker serializes inference anyway (docs/prd/command-queue.md).
  const enqueue = useRef(createQueue()).current;

  function run(q: string) {
    const text = q.trim();
    if (!text) return;
    setQuery("");
    setPending((n) => n + 1);
    void enqueue(() => exec(text))
      .catch((err) => toast.error(`Command failed: ${String(err)}`))
      .finally(() => setPending((n) => n - 1));
  }

  async function exec(text: string) {
    void loadNeedle((p) => {
      const pct = Math.round((p.loaded / p.total) * 100);
      setLoadPct(pct === 100 ? null : pct);
    }).catch(() => setLoadPct(null));
    const call = await wasmNeedle.infer(text);
    setLoadPct(null);
    setLastCall(call);
    if (call.tool === "show_help") {
      toast.info(
        'Try "Add buy milk", "Rename practice guitar to practice piano", "I didn\'t finish buy milk", or "What\'s due Friday?".',
      );
      return;
    }
    const data = (await callApi(call)) as TaskData | null; // mutations happen here
    if (data?.note) toast(data.note);
    const nextFilter =
      call.tool === "show_tasks"
        ? call.args.filter || "all"
        : filterRef.current;
    filterRef.current = nextFilter;
    // Render the view callApi returned — it already reflects the mutation
    // and honors args (like show_tasks' due) that a refetch would drop.
    setSpec(tasksSpec(data ?? (await fetchTasks(nextFilter))));
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="flex-1 overflow-y-auto pt-4 pb-4">
        {spec && (
          <JSONUIProvider registry={registry}>
            <Renderer spec={spec} registry={registry} />
          </JSONUIProvider>
        )}
        <div className="text-muted-foreground mt-4 flex flex-wrap gap-2 text-xs">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => void run(s)}
              className="border-border hover:text-foreground rounded-full border px-3 py-1 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-1 pt-1">
        {loadPct !== null ? (
          <p className="text-muted-foreground truncate font-mono text-xs">
            ▸ downloading model… {loadPct}%
          </p>
        ) : (
          lastCall && (
            <p className="text-muted-foreground truncate font-mono text-xs">
              ▸ {toolCallLabel(lastCall)}
            </p>
          )
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run(query);
          }}
          className="flex gap-2"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Tell me what to do with your tasks…"
            className="border-input bg-background focus-visible:ring-ring flex-1 rounded-full border px-4 py-2 text-sm shadow-sm focus-visible:ring-1 focus-visible:outline-none"
          />
          <button
            type="submit"
            disabled={!query.trim()}
            className="bg-primary text-primary-foreground rounded-full px-4 py-2 text-sm font-medium shadow disabled:opacity-50"
          >
            {pending > 0 ? `Send (${pending})` : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
}
