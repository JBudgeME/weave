import { describe, expect, test } from "bun:test";

import { mockNeedle } from "./model";

const CASES: Array<[string, string, Record<string, string>?]> = [
  ["Add buy milk", "add_task", { title: "buy milk" }],
  ["Add buy milk to my tasks", "add_task", { title: "buy milk" }],
  ["Create a task walk the dog", "add_task", { title: "walk the dog" }],
  ["Mark buy milk as done", "complete_task", { title: "buy milk" }],
  ["Complete buy milk", "complete_task", { title: "buy milk" }],
  ["Delete buy milk", "delete_task", { title: "buy milk" }],
  ["Remove the buy milk task", "delete_task"],
  ["Show my tasks", "show_tasks", { filter: "all" }],
  ["Show open tasks", "show_tasks", { filter: "open" }],
  ["What have I done?", "show_tasks", { filter: "done" }],
  [
    "Rename buy milk to buy oat milk",
    "edit_task",
    { title: "buy milk", new_title: "buy oat milk" },
  ],
  [
    "Move buy milk to Friday",
    "edit_task",
    { title: "buy milk", new_due: "Friday" },
  ],
  ["I didn't finish buy milk", "uncomplete_task", { title: "buy milk" }],
  ["Uncheck buy milk", "uncomplete_task", { title: "buy milk" }],
  ["What's due Friday?", "show_tasks", { due: "Friday" }],
  ["blorp", "show_help"],
];

describe("mockNeedle routing", () => {
  for (const [query, tool, args] of CASES) {
    test(`"${query}" → ${tool}`, async () => {
      const call = await mockNeedle.infer(query);
      expect(call.tool).toBe(tool);
      if (args) expect(call.args).toMatchObject(args);
    });
  }
});
