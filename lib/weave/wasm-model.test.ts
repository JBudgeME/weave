import { describe, expect, test } from "bun:test";

import { toToolCall } from "./wasm-model";

describe("toToolCall", () => {
  test("parses a tool call with args", () => {
    expect(
      toToolCall(
        '[{"name":"add_task","arguments":{"title":"pick up groceries","due":"Saturday"}}]',
      ),
    ).toEqual({
      tool: "add_task",
      args: { title: "pick up groceries", due: "Saturday" },
      source: "wasm",
    });
  });

  test("stringifies non-string arg values", () => {
    expect(
      toToolCall('[{"name":"show_tasks","arguments":{"filter":1}}]'),
    ).toEqual({ tool: "show_tasks", args: { filter: "1" }, source: "wasm" });
  });

  test("empty or malformed output falls back to show_help", () => {
    expect(toToolCall("[]").tool).toBe("show_help");
    expect(toToolCall("garbage").tool).toBe("show_help");
  });
});
