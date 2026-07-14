import { describe, expect, test } from "bun:test";

import type { StoredTask } from "./store";
import { tasksSpec } from "./templates";

const task = (over: Partial<StoredTask> = {}): StoredTask => ({
  title: "buy milk",
  status: "open",
  due: "—",
  ...over,
});

describe("tasksSpec", () => {
  test("empty list renders the empty-state text, no table", () => {
    const spec = tasksSpec({ filter: "all", tasks: [] });
    expect(spec.elements.empty).toBeDefined();
    expect(spec.elements.table).toBeUndefined();
    expect(spec.elements.progress.props.label).toBe("0 of 0 done");
    expect(spec.elements.progress.props.value).toBe(0);
  });

  test("rows map title/status/due and done count drives progress", () => {
    const spec = tasksSpec({
      filter: "all",
      tasks: [
        task(),
        task({ title: "walk the dog", status: "done", due: "Fri" }),
      ],
    });
    expect(spec.elements.table.props.rows).toEqual([
      ["buy milk", "◻️ open", "—"],
      ["walk the dog", "✅ done", "Fri"],
    ]);
    expect(spec.elements.progress.props.value).toBe(50);
    expect(spec.elements.progress.props.label).toBe("1 of 2 done");
  });

  test("filter label (including due filter) lands in the card description", () => {
    const spec = tasksSpec({ filter: "all · due Friday", tasks: [task()] });
    expect(spec.elements.root.props.description).toBe(
      "Filter: all · due Friday",
    );
  });
});
