/**
 * Template layer: ToolCall + API data → json-render Spec.
 * The model never generates structure; these functions own the tree,
 * so the output is valid by construction.
 */

import type { Spec, UIElement } from "@json-render/core";
import type { TaskData } from "./api";

type Elements = Record<string, UIElement>;

function el(
  type: string,
  props: Record<string, unknown>,
  children?: string[],
): UIElement {
  return { type, props, ...(children ? { children } : {}) };
}

export function tasksSpec(data: TaskData): Spec {
  const done = data.tasks.filter((t) => t.status === "done").length;
  const stackChildren: string[] = ["progress"];
  const elements: Elements = {
    root: el(
      "Card",
      {
        title: "Tasks",
        description: `Filter: ${data.filter}`,
        maxWidth: "full",
      },
      ["stack"],
    ),
    progress: el("Progress", {
      value: data.tasks.length ? (done / data.tasks.length) * 100 : 0,
      label: `${done} of ${data.tasks.length} done`,
    }),
  };
  if (data.tasks.length === 0) {
    elements.empty = el("Text", {
      text: 'Nothing here. Try "add buy milk".',
      variant: "muted",
    });
    stackChildren.push("empty");
  } else {
    elements.table = el("Table", {
      columns: ["Task", "Status", "Due"],
      rows: data.tasks.map((t) => [
        t.title,
        t.status === "done" ? "✅ done" : "◻️ open",
        t.due,
      ]),
    });
    stackChildren.push("table");
  }
  elements.stack = el(
    "Stack",
    { direction: "vertical", gap: "lg", align: "stretch" },
    stackChildren,
  );
  return { root: "root", elements };
}
