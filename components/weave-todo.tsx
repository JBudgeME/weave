"use client";

import { useEffect, useRef, useState } from "react";
import type { Spec } from "@json-render/core";
import { JSONUIProvider, Renderer } from "@json-render/react";
import { toast } from "sonner";

import { callApi, type TaskData } from "@/lib/weave/api";
import { CommandSession } from "@/lib/weave/command-session";
import { type ToolCall } from "@/lib/weave/model";
import { loadNeedle } from "@/lib/weave/wasm-model";
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
  const [query, setQuery] = useState("");
  const [pending, setPending] = useState(0);
  const [lastCall, setLastCall] = useState<ToolCall | null>(null);
  const [loadPct, setLoadPct] = useState<number | null>(null);

  useEffect(() => {
    void (
      callApi({
        tool: "show_tasks",
        args: { filter: "all" },
      }) as Promise<TaskData>
    ).then((d) => setSpec(tasksSpec(d)));
  }, []);

  // The command pipeline (model → API → spec, FIFO queue, filter state)
  // lives in CommandSession; the component only renders its results.
  const session = useRef(new CommandSession()).current;

  function run(q: string) {
    const text = q.trim();
    if (!text) return;
    setQuery("");
    setPending((n) => n + 1);
    void loadNeedle((p) => {
      const pct = Math.round((p.loaded / p.total) * 100);
      setLoadPct(pct === 100 ? null : pct);
    }).catch(() => setLoadPct(null));
    void session
      .run(text)
      .then((r) => {
        setLoadPct(null);
        setLastCall(r.call);
        if (r.call.tool === "show_help") {
          if (r.note) toast.info(r.note);
          return;
        }
        if (r.note) toast(r.note);
        if (r.spec) setSpec(r.spec);
      })
      .catch((err) => toast.error(`Command failed: ${String(err)}`))
      .finally(() => setPending((n) => n - 1));
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
