/**
 * The model seam. Anything that turns a natural-language query into a
 * ToolCall satisfies UIModel — today a keyword mock, later Needle in WASM.
 */

export interface ToolCall {
  tool: string;
  args: Record<string, string>;
  /** Which model produced this call ("needle" | "mock"). */
  source?: string;
}

export interface UIModel {
  infer(query: string): Promise<ToolCall>;
}

/** Strip verb phrases and list suffixes to get a bare task title. */
function extractTitle(query: string): string {
  return query
    .replace(
      /^\s*(?:please\s+)?(?:add|create|new|complete|finish|delete|remove|check off|mark)\s+(?:a\s+)?(?:task\s+)?/i,
      "",
    )
    .replace(
      /\s+(?:to|from|off)\s+(?:my\s+)?(?:tasks?|to-?do)(?:\s+list)?/i,
      "",
    )
    .replace(/\s+as\s+done\s*$/i, "")
    .replace(/[?.!]\s*$/, "")
    .trim();
}

/**
 * Mock Needle: keyword intent matching with fake inference latency.
 * ponytail: regex intent picker — swap for real Needle inference at this seam.
 */
export const mockNeedle: UIModel = {
  async infer(query: string): Promise<ToolCall> {
    await new Promise((r) => setTimeout(r, 350)); // pretend to think

    const q = query.toLowerCase();
    const mock = (
      tool: string,
      args: Record<string, string> = {},
    ): ToolCall => ({ tool, args, source: "mock" });

    if (/\bmark\b.*\bdone\b/.test(q) || /\bcheck off\b/.test(q)) {
      return mock("complete_task", { title: extractTitle(query) });
    }
    if (/^\s*(?:please\s+)?(?:complete|finish)\b/.test(q)) {
      return mock("complete_task", { title: extractTitle(query) });
    }
    if (
      /\b(?:didn'?t|did not|haven'?t|have not)\s+(?:finish|complete)\b/.test(
        q,
      ) ||
      /\b(?:uncheck|reopen|un-?complete)\b/.test(q)
    ) {
      const title = query
        .replace(
          /^\s*(?:i\s+)?(?:didn'?t|did not|haven'?t|have not)\s+(?:finish|complete)\s+/i,
          "",
        )
        .replace(/^\s*(?:please\s+)?(?:uncheck|reopen|un-?complete)\s+/i, "")
        .replace(/[?.!]\s*$/, "")
        .trim();
      return mock("uncomplete_task", { title });
    }
    const renameMatch = query.match(
      /^\s*(?:please\s+)?rename\s+(.+?)\s+to\s+(.+?)\s*[?.!]?\s*$/i,
    );
    if (renameMatch) {
      return mock("edit_task", {
        title: renameMatch[1],
        new_title: renameMatch[2],
      });
    }
    const moveMatch = query.match(
      /^\s*(?:please\s+)?move\s+(.+?)\s+to\s+(.+?)\s*[?.!]?\s*$/i,
    );
    if (moveMatch) {
      return mock("edit_task", {
        title: moveMatch[1],
        new_due: moveMatch[2],
      });
    }
    if (/^\s*(?:please\s+)?(?:add|create|new)\b/.test(q)) {
      return mock("add_task", { title: extractTitle(query) });
    }
    if (/^\s*(?:please\s+)?(?:delete|remove|drop)\b/.test(q)) {
      return mock("delete_task", { title: extractTitle(query) });
    }
    if (/\b(show|list|view|what)\b/.test(q) || /\btasks?\b/.test(q)) {
      const args: Record<string, string> = {};
      const dueMatch = query.match(/\bdue\s+(?:on\s+)?(\w+)/i);
      if (dueMatch) {
        args.due = dueMatch[1];
      } else {
        args.filter = /\bdone\b/.test(q)
          ? "done"
          : /\b(open|pending|left|remaining)\b/.test(q)
            ? "open"
            : "all";
      }
      return mock("show_tasks", args);
    }
    return mock("show_help");
  },
};

/**
 * Real Needle via the /api/infer route (CLI on the server). Falls back to
 * the mock when the route errors or needle isn't installed.
 */
export const needle: UIModel = {
  async infer(query: string): Promise<ToolCall> {
    try {
      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error(`infer ${res.status}`);
      return (await res.json()) as ToolCall;
    } catch {
      return mockNeedle.infer(query);
    }
  },
};
