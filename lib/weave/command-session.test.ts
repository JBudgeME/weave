import { beforeEach, describe, expect, test } from "bun:test";
import type { Spec, UIElement } from "@json-render/core";

import { CommandSession, HELP_NOTE } from "./command-session";
import type { ToolCall, UIModel } from "./model";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Fake model: scripted text → ToolCall, optional per-call delay. */
function fakeModel(
  script: (query: string) => ToolCall,
  delayMs: (query: string) => number = () => 0,
): UIModel {
  return {
    async infer(query: string): Promise<ToolCall> {
      await sleep(delayMs(query));
      return script(query);
    },
  };
}

function cardDescription(spec: Spec | undefined): string {
  const root = spec?.elements.root as UIElement;
  return String(root.props.description);
}

function tableTitles(spec: Spec | undefined): string[] {
  const table = spec?.elements.table as UIElement;
  return (table.props.rows as string[][]).map((r) => r[0]);
}

// happy-dom gives us a real localStorage (see api.test.ts) — the session runs
// against the REAL callApi + store; only the model is faked.
beforeEach(() => {
  window.localStorage.clear();
});

describe("CommandSession", () => {
  test("Due filter is tracked across sequential commands", async () => {
    const session = new CommandSession(
      fakeModel((q) => {
        if (q.startsWith("add"))
          return { tool: "add_task", args: { title: q.slice(4) } };
        if (q.startsWith("show"))
          return { tool: "show_tasks", args: { filter: "done" } };
        return { tool: "complete_task", args: { title: q.slice(9) } };
      }),
    );
    expect(session.filter).toBe("all");
    await session.run("add buy milk");
    expect(session.filter).toBe("all");
    await session.run("show done tasks");
    expect(session.filter).toBe("done");
    // Mutations keep the last-selected filter — the old stale-closure trap.
    const r = await session.run("complete buy milk");
    expect(session.filter).toBe("done");
    expect(r.note).toBe('Marked "buy milk" done.');
  });

  test("renders the view callApi returned — show_tasks' due arg survives", async () => {
    const session = new CommandSession(
      fakeModel(() => ({ tool: "show_tasks", args: { due: "Mon" } })),
    );
    const r = await session.run("what's due Monday?");
    // A refetch would drop the due arg and show every task; the returned
    // view keeps the due-filtered rows and label.
    expect(cardDescription(r.spec)).toBe("Filter: all · due Mon");
    expect(tableTitles(r.spec)).toEqual(["Strip greenfield branding"]);
  });

  test("show_help returns the hint note only — no spec, filter untouched", async () => {
    const session = new CommandSession(
      fakeModel(() => ({ tool: "show_help", args: {} })),
    );
    const r = await session.run("gibberish");
    expect(r.call.tool).toBe("show_help");
    expect(r.note).toBe(HELP_NOTE);
    expect(r.spec).toBeUndefined();
    expect(session.filter).toBe("all");
  });

  test("overlapping run() calls resolve FIFO even when the first is slower", async () => {
    const session = new CommandSession(
      fakeModel(
        (q) => ({ tool: "add_task", args: { title: q.slice(4) } }),
        (q) => (q.includes("slow") ? 40 : 0),
      ),
    );
    const done: string[] = [];
    const a = session.run("add slow").then(() => done.push("slow"));
    const b = session.run("add fast").then((r) => {
      done.push("fast");
      return r;
    });
    const [, second] = await Promise.all([a, b]);
    expect(done).toEqual(["slow", "fast"]);
    // Second command's view reflects both adds, in submission order.
    expect(tableTitles(second.spec)).toContain("slow");
    expect(tableTitles(second.spec)[0]).toBe("fast");
  });
});
