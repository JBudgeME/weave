import { beforeEach, describe, expect, mock, test } from "bun:test";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// Mock the inference layer before importing the component: tests drive the
// UI + queue wiring, not the model.
const inferLog: string[] = [];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Keep the real exports (toToolCall is tested elsewhere and mock.module is
// process-global in bun) — only the inference entry points are replaced.
const real = await import("@/lib/weave/wasm-model");

mock.module("@/lib/weave/wasm-model", () => ({
  ...real,
  loadNeedle: async () => ({}),
  wasmNeedle: {
    async infer(query: string) {
      // First command is slow, later ones fast — FIFO must still hold.
      await sleep(inferLog.length === 0 ? 60 : 1);
      inferLog.push(query);
      if (query.startsWith("Add ")) {
        return { tool: "add_task", args: { title: query.slice(4) } };
      }
      if (query.startsWith("Complete ")) {
        return { tool: "complete_task", args: { title: query.slice(9) } };
      }
      return { tool: "show_tasks", args: {} };
    },
  },
}));

const { WeaveTodo } = await import("./weave-todo");

function submit(text: string) {
  const input = screen.getByPlaceholderText<HTMLInputElement>(/tell me what/i);
  fireEvent.change(input, { target: { value: text } });
  fireEvent.submit(input.closest("form")!);
}

describe("WeaveTodo command queue", () => {
  beforeEach(() => {
    window.localStorage.clear();
    inferLog.length = 0;
  });

  test("input never disables and back-to-back commands run FIFO", async () => {
    render(<WeaveTodo />);
    const input = screen.getByPlaceholderText(/tell me what/i);

    submit("Add buy milk");
    expect(input).toBeEnabled(); // no lock while the slow command runs
    submit("Complete buy milk");
    expect(input).toBeEnabled();

    // Both ran, in submission order — so the add landed before the complete.
    await waitFor(() => expect(inferLog.length).toBe(2), { timeout: 3000 });
    expect(inferLog).toEqual(["Add buy milk", "Complete buy milk"]);
    await waitFor(() =>
      expect(screen.getByText(/✅ done/)).toBeInTheDocument(),
    );
  });

  test("pending count surfaces on the Send button while queued", async () => {
    render(<WeaveTodo />);
    submit("Add buy milk");
    submit("Add call mom");
    expect(
      screen.getByRole("button", { name: /send \(2\)/i }),
    ).toBeInTheDocument();
    await waitFor(
      () =>
        expect(
          screen.getByRole("button", { name: /^send$/i }),
        ).toBeInTheDocument(),
      { timeout: 3000 },
    );
    expect(inferLog).toEqual(["Add buy milk", "Add call mom"]);
  });
});
