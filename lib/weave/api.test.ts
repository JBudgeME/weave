import { beforeEach, describe, expect, test } from "bun:test";

import { callApi, dueMatches } from "./api";

// happy-dom gives us a real localStorage, so the store (store.ts) persists
// across calls just like in the browser — reset it before each test so
// cases don't see mutations from prior tests.
beforeEach(() => {
  window.localStorage.clear();
});

describe("callApi", () => {
  test("edit_task renames a task", async () => {
    const data = await callApi({
      tool: "edit_task",
      args: { title: "port needle", new_title: "port needle to wasm2" },
    });
    expect(data?.note).toBe('Updated "Port Needle to WASM".');
  });

  test("edit_task reschedules a task", async () => {
    const data = await callApi({
      tool: "edit_task",
      args: { title: "port needle", new_due: "Tuesday" },
    });
    expect(data?.note).toBe('Updated "Port Needle to WASM".');
  });

  test("edit_task can rename and reschedule together", async () => {
    const data = await callApi({
      tool: "edit_task",
      args: {
        title: "port needle",
        new_title: "port needle to wasm2",
        new_due: "Tuesday",
      },
    });
    expect(data?.note).toBe('Updated "Port Needle to WASM".');
  });

  test("edit_task on unknown title is a no-op with a not-found note", async () => {
    const data = await callApi({
      tool: "edit_task",
      args: { title: "nonexistent", new_title: "x" },
    });
    expect(data?.note).toBe('No task matching "nonexistent".');
  });

  test("uncomplete_task reopens a done task", async () => {
    const data = await callApi({
      tool: "uncomplete_task",
      args: { title: "strip greenfield" },
    });
    expect(data?.note).toBe('Reopened "Strip greenfield branding".');
  });

  test("uncomplete_task on unknown title is a no-op with a not-found note", async () => {
    const data = await callApi({
      tool: "uncomplete_task",
      args: { title: "nonexistent" },
    });
    expect(data?.note).toBe('No task matching "nonexistent".');
  });

  test("show_tasks with due filters by substring", async () => {
    const data = await callApi({
      tool: "show_tasks",
      args: { due: "Mon" },
    });
    expect(data?.tasks.map((t) => t.title)).toEqual([
      "Strip greenfield branding",
    ]);
  });

  test("dueMatches: a task due 'Fri' is found by querying 'Friday' (alias)", () => {
    expect(dueMatches("Fri", "Friday")).toBe(true);
  });

  test("dueMatches: unrelated days don't match", () => {
    expect(dueMatches("Fri", "Monday")).toBe(false);
  });

  test("show_tasks with due matches the Monday alias against 'Mon'", async () => {
    const data = await callApi({
      tool: "show_tasks",
      args: { due: "Monday" },
    });
    expect(data?.tasks.map((t) => t.title)).toEqual([
      "Strip greenfield branding",
    ]);
  });
});
