/**
 * Task API. Reads/writes go through the localStorage store; swap for a
 * real backend later — the templates only see the returned shapes.
 */

import type { ToolCall } from "./model";
import { getStore, updateStore, type StoredTask } from "./store";

export interface TaskData {
  filter: string;
  note?: string;
  tasks: StoredTask[];
}

export type ApiData = TaskData | null;

// ponytail: substring+alias only, real date parsing if dues become dates
const DAY_ALIASES: Record<string, string> = {
  mon: "monday",
  tue: "tuesday",
  tues: "tuesday",
  wed: "wednesday",
  thu: "thursday",
  thur: "thursday",
  thurs: "thursday",
  fri: "friday",
  sat: "saturday",
  sun: "sunday",
};

export function dueMatches(due: string, query: string): boolean {
  const d = due.toLowerCase();
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (d.includes(q)) return true;
  // either side may be a short form ("Fri") of the other's full name
  const dFull = DAY_ALIASES[d] ?? d;
  const qFull = DAY_ALIASES[q] ?? q;
  return dFull === qFull || dFull.includes(qFull) || qFull.includes(dFull);
}

function tasksView(filter: string, note?: string, due?: string): TaskData {
  const { tasks } = getStore();
  let list =
    filter === "all" ? tasks : tasks.filter((t) => t.status === filter);
  if (due) list = list.filter((t) => dueMatches(t.due, due));
  // Surface the due filter in the card's "Filter:" label so the header
  // matches the (filtered) rows actually shown.
  return { filter: due ? `${filter} · due ${due}` : filter, note, tasks: list };
}

function findTask(tasks: StoredTask[], title: string): StoredTask | undefined {
  const q = title.toLowerCase();
  return tasks.find((t) => t.title.toLowerCase().includes(q));
}

export async function callApi(call: ToolCall): Promise<ApiData> {
  await new Promise((r) => setTimeout(r, 150)); // pretend network
  const { args } = call;

  switch (call.tool) {
    case "show_tasks":
      return tasksView(args.filter || "all", undefined, args.due);
    case "add_task": {
      const title = args.title?.trim();
      if (!title) return tasksView("all", "No task title given.");
      updateStore((s) => {
        s.tasks.unshift({ title, status: "open", due: args.due || "—" });
      });
      return tasksView("all", `Added "${title}".`);
    }
    case "complete_task": {
      let hit: string | undefined;
      updateStore((s) => {
        const t = findTask(s.tasks, args.title ?? "");
        if (t) {
          t.status = "done";
          hit = t.title;
        }
      });
      return tasksView(
        "all",
        hit ? `Marked "${hit}" done.` : `No task matching "${args.title}".`,
      );
    }
    case "edit_task": {
      let hit: string | undefined;
      updateStore((s) => {
        const t = findTask(s.tasks, args.title ?? "");
        if (t) {
          hit = t.title;
          if (args.new_title?.trim()) t.title = args.new_title.trim();
          if (args.new_due?.trim()) t.due = args.new_due.trim();
        }
      });
      return tasksView(
        "all",
        hit ? `Updated "${hit}".` : `No task matching "${args.title}".`,
      );
    }
    case "uncomplete_task": {
      let hit: string | undefined;
      updateStore((s) => {
        const t = findTask(s.tasks, args.title ?? "");
        if (t) {
          t.status = "open";
          hit = t.title;
        }
      });
      return tasksView(
        "all",
        hit ? `Reopened "${hit}".` : `No task matching "${args.title}".`,
      );
    }
    case "delete_task": {
      let hit: string | undefined;
      updateStore((s) => {
        const t = findTask(s.tasks, args.title ?? "");
        if (t) {
          hit = t.title;
          s.tasks = s.tasks.filter((x) => x !== t);
        }
      });
      return tasksView(
        "all",
        hit ? `Deleted "${hit}".` : `No task matching "${args.title}".`,
      );
    }
    default:
      return null;
  }
}
