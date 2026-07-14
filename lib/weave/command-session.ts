/**
 * Command pipeline: NL text → model ToolCall → API call → renderable Spec.
 * Owns the Due-filter state and the FIFO queue, so commands can never
 * execute out of order and queued commands always see the latest filter
 * (the stale-closure bug the component's filterRef used to work around).
 */

import type { Spec } from "@json-render/core";

import { callApi, type TaskData } from "./api";
import type { ToolCall, UIModel } from "./model";
import { createQueue } from "./queue";
import { tasksSpec } from "./templates";
import { wasmNeedle } from "./wasm-model";

export interface CommandResult {
  call: ToolCall;
  /** Absent only for show_help (toast-only, view untouched). */
  spec?: Spec;
  note?: string;
}

export const HELP_NOTE =
  'Try "Add buy milk", "Rename practice guitar to practice piano", "I didn\'t finish buy milk", or "What\'s due Friday?".';

export class CommandSession {
  #filter = "all";
  // Commands queue FIFO so the input never locks while one is running; the
  // worker serializes inference anyway (docs/prd/command-queue.md).
  #enqueue = createQueue();

  constructor(
    private model: UIModel = wasmNeedle,
    private api: typeof callApi = callApi,
  ) {}

  get filter(): string {
    return this.#filter;
  }

  /** Enqueues internally — out-of-order execution is impossible. */
  run(text: string): Promise<CommandResult> {
    return this.#enqueue(() => this.exec(text));
  }

  async #refetch(): Promise<TaskData> {
    return (await this.api({
      tool: "show_tasks",
      args: { filter: this.#filter },
    })) as TaskData;
  }

  private async exec(text: string): Promise<CommandResult> {
    const call = await this.model.infer(text);
    if (call.tool === "show_help") return { call, note: HELP_NOTE };
    const data = (await this.api(call)) as TaskData | null; // mutations happen here
    this.#filter =
      call.tool === "show_tasks" ? call.args.filter || "all" : this.#filter;
    // Render the view the API returned — it already reflects the mutation
    // and honors args (like show_tasks' due) that a refetch would drop.
    // Refetch only when the API returned nothing.
    const view = data ?? (await this.#refetch());
    return { call, spec: tasksSpec(view), note: data?.note };
  }
}
