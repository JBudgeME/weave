/**
 * Deterministic slot-fill generator for Needle finetuning over the frozen
 * 4-tool catalog. Emits needle JSONL ({query, tools, answers} with
 * JSON-encoded tools/answers) as train + held-out eval splits, plus a
 * regression set of documented live misses (README "Known quirks").
 *
 * Run: bun scripts/gen-finetune-data.ts  → writes finetune/{train,eval,regression}.jsonl
 */
import { mkdirSync, writeFileSync } from "node:fs";

import { TOOLS } from "../lib/weave/tools";

// ponytail: mulberry32 — seeded so the dataset is reproducible byte-for-byte.
function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(42);
const pick = <T>(arr: readonly T[]): T => arr[Math.floor(rand() * arr.length)];

// Multi-word titles dominate — the documented failure is space-squashing.
const TITLES = [
  "pick up groceries",
  "buy milk",
  "call mom",
  "walk the dog",
  "water the plants",
  "pay the rent",
  "do the laundry",
  "clean the garage",
  "email the landlord",
  "book a dentist appointment",
  "renew car insurance",
  "take out the trash",
  "fix the leaky faucet",
  "wash the car",
  "return library books",
  "schedule oil change",
  "buy birthday gift for Sam",
  "mow the lawn",
  "charge the drill batteries",
  "back up the laptop",
  "print boarding passes",
  "pack for the trip",
  "pick up dry cleaning",
  "refill the dog food",
  "call the plumber",
  "update the resume",
  "send the invoice",
  "order new filters",
  "plan the weekend menu",
  "read chapter five",
  "practice guitar",
  "stretch for ten minutes",
  "check the mail",
  "defrost the chicken",
  "sign the permission slip",
  "buy stamps",
  "organize the closet",
  "test the smoke alarms",
  "write thank you notes",
  "review the budget",
] as const;

const DUES = [
  "Saturday",
  "Sunday",
  "Monday",
  "Tuesday",
  "Friday",
  "tomorrow",
  "tonight",
  "next week",
  "this weekend",
  "Fri",
  "Thursday",
  "today",
] as const;

type Example = { query: string; tool: string; args: Record<string, string> };

function addExamples(n: number): Example[] {
  const verbs = [
    (t: string) => `Add ${t}`,
    (t: string) => `add ${t} to my list`,
    (t: string) => `Add a task to ${t}`,
    (t: string) => `Create a task: ${t}`,
    (t: string) => `Remember to ${t}`,
    (t: string) => `New task: ${t}`,
    (t: string) => `Put ${t} on my todo list`,
    (t: string) => `I need to ${t}`,
    (t: string) => `Don't let me forget to ${t}`,
    (t: string) => `add ${t}`,
  ];
  const dueFmts = [
    (d: string) => `on ${d}`,
    (d: string) => `by ${d}`,
    (d: string) => `due ${d}`,
    (d: string) => `${d}`,
  ];
  const out: Example[] = [];
  for (let i = 0; i < n; i++) {
    const title = pick(TITLES);
    const verb = pick(verbs);
    if (rand() < 0.5) {
      const due = pick(DUES);
      out.push({
        query: `${verb(title)} ${pick(dueFmts)(due)}`,
        tool: "add_task",
        args: { title, due },
      });
    } else {
      out.push({ query: verb(title), tool: "add_task", args: { title } });
    }
  }
  return out;
}

function completeExamples(n: number): Example[] {
  const verbs = [
    (t: string) => `Mark ${t} as done`,
    (t: string) => `mark ${t} done`,
    (t: string) => `Check off ${t}`,
    (t: string) => `Complete ${t}`,
    (t: string) => `Finish ${t}`,
    (t: string) => `I finished ${t}`,
    (t: string) => `${t} is done`,
    (t: string) => `Done with ${t}`,
    (t: string) => `I already did ${t}`,
    (t: string) => `cross off ${t}`,
  ];
  return Array.from({ length: n }, () => {
    const title = pick(TITLES);
    return {
      query: pick(verbs)(title),
      tool: "complete_task",
      args: { title },
    };
  });
}

function deleteExamples(n: number): Example[] {
  const verbs = [
    (t: string) => `Delete ${t}`,
    (t: string) => `delete the ${t} task`,
    (t: string) => `Remove ${t}`,
    (t: string) => `remove ${t} from my list`,
    (t: string) => `Get rid of ${t}`,
    (t: string) => `Drop ${t}`,
    (t: string) => `Scratch ${t} off the list`,
    (t: string) => `Cancel ${t}`,
    (t: string) => `I don't need to ${t} anymore`,
    (t: string) => `Take ${t} off my todo list`,
  ];
  return Array.from({ length: n }, () => {
    const title = pick(TITLES);
    return { query: pick(verbs)(title), tool: "delete_task", args: { title } };
  });
}

function editExamples(n: number): Example[] {
  const renames = [
    (t: string, nt: string) => `Rename ${t} to ${nt}`,
    (t: string, nt: string) => `rename the ${t} task to ${nt}`,
    (t: string, nt: string) => `Change ${t} to ${nt}`,
    (t: string, nt: string) => `Update ${t} to say ${nt}`,
    (t: string, nt: string) => `Make ${t} read ${nt}`,
  ];
  const reschedules = [
    (t: string, d: string) => `Move ${t} to ${d}`,
    (t: string, d: string) => `Push ${t} to ${d}`,
    (t: string, d: string) => `Reschedule ${t} for ${d}`,
    (t: string, d: string) => `Change the due date of ${t} to ${d}`,
    (t: string, d: string) => `Make ${t} due ${d}`,
  ];
  return Array.from({ length: n }, (): Example => {
    const title = pick(TITLES);
    if (rand() < 0.5) {
      let newTitle = pick(TITLES);
      while (newTitle === title) newTitle = pick(TITLES);
      return {
        query: pick(renames)(title, newTitle),
        tool: "edit_task",
        args: { title, new_title: newTitle },
      };
    }
    const due = pick(DUES);
    return {
      query: pick(reschedules)(title, due),
      tool: "edit_task",
      args: { title, new_due: due },
    };
  });
}

function uncompleteExamples(n: number): Example[] {
  const verbs = [
    (t: string) => `Uncheck ${t}`,
    (t: string) => `uncheck the ${t} task`,
    (t: string) => `Reopen ${t}`,
    (t: string) => `Mark ${t} as not done`,
    (t: string) => `${t} isn't actually done`,
    (t: string) => `I didn't finish ${t}`,
    (t: string) => `Undo completing ${t}`,
    (t: string) => `Put ${t} back on my list`,
    (t: string) => `Un-complete ${t}`,
    (t: string) => `Actually ${t} still needs doing`,
  ];
  return Array.from({ length: n }, () => {
    const title = pick(TITLES);
    return {
      query: pick(verbs)(title),
      tool: "uncomplete_task",
      args: { title },
    };
  });
}

function showDueExamples(n: number): Example[] {
  const verbs = [
    (d: string) => `What's due ${d}?`,
    (d: string) => `what's due ${d}`,
    (d: string) => `Show tasks due ${d}`,
    (d: string) => `show me what's due ${d}`,
    (d: string) => `List tasks due ${d}`,
    (d: string) => `What do I have due ${d}?`,
    (d: string) => `Anything due ${d}?`,
    (d: string) => `Show everything due ${d}`,
  ];
  return Array.from({ length: n }, () => {
    const due = pick(DUES);
    return { query: pick(verbs)(due), tool: "show_tasks", args: { due } };
  });
}

function showExamples(n: number): Example[] {
  const byFilter: Record<string, string[]> = {
    all: [
      "Show my tasks",
      "show all tasks",
      "List my tasks",
      "What's on my list?",
      "Show everything",
      "View my todo list",
      "What are my tasks?",
      "show the whole list",
      "Display all my tasks",
      "what do I have on my plate",
    ],
    open: [
      "Show open tasks",
      "What's left?",
      "list open tasks",
      "What do I still need to do?",
      "Show remaining tasks",
      "what's still pending",
      "Show my unfinished tasks",
      "What's outstanding?",
      "show what's left to do",
      "View pending tasks",
    ],
    done: [
      "Show done tasks",
      "show completed tasks",
      "What have I finished?",
      "List finished tasks",
      "Show what I've done",
      "view my completed tasks",
      "What did I get done?",
      "Show my finished items",
      "list everything I completed",
      "What's already done?",
    ],
  };
  return Array.from({ length: n }, () => {
    const filter = pick(["all", "open", "done"] as const);
    return {
      query: pick(byFilter[filter]),
      tool: "show_tasks",
      args: { filter },
    };
  });
}

// Documented live misses — held out entirely from training as a regression set.
const REGRESSION: Example[] = [
  {
    query: "Add pick up groceries on Saturday",
    tool: "add_task",
    args: { title: "pick up groceries", due: "Saturday" },
  },
  {
    query: "Add pick up groceries",
    tool: "add_task",
    args: { title: "pick up groceries" },
  },
  { query: "Add buy milk", tool: "add_task", args: { title: "buy milk" } },
  {
    query: "Mark buy milk as done",
    tool: "complete_task",
    args: { title: "buy milk" },
  },
  { query: "Show open tasks", tool: "show_tasks", args: { filter: "open" } },
  {
    query: "Remove call mom from my list",
    tool: "delete_task",
    args: { title: "call mom" },
  },
  {
    query: "Add water the plants tomorrow",
    tool: "add_task",
    args: { title: "water the plants", due: "tomorrow" },
  },
  {
    query: "Finish walk the dog",
    tool: "complete_task",
    args: { title: "walk the dog" },
  },
  {
    query: "Rename practice guitar to practice piano",
    tool: "edit_task",
    args: { title: "practice guitar", new_title: "practice piano" },
  },
  {
    query: "Move buy stamps to Friday",
    tool: "edit_task",
    args: { title: "buy stamps", new_due: "Friday" },
  },
  {
    query: "I didn't finish pick up groceries",
    tool: "uncomplete_task",
    args: { title: "pick up groceries" },
  },
  {
    query: "What's due Saturday?",
    tool: "show_tasks",
    args: { due: "Saturday" },
  },
];

const toolsJson = JSON.stringify(TOOLS);
const toLine = (e: Example) =>
  JSON.stringify({
    query: e.query,
    tools: toolsJson,
    answers: JSON.stringify([{ name: e.tool, arguments: e.args }]),
  });

const all = [
  ...addExamples(800),
  ...completeExamples(600),
  ...deleteExamples(600),
  ...showExamples(600),
  ...editExamples(700),
  ...uncompleteExamples(600),
  ...showDueExamples(400),
];
// Deterministic shuffle, then hold out 10% for eval.
for (let i = all.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  [all[i], all[j]] = [all[j], all[i]];
}
const cut = Math.floor(all.length * 0.9);
const train = all.slice(0, cut);
const evalSet = all.slice(cut);

mkdirSync("finetune", { recursive: true });
writeFileSync("finetune/train.jsonl", train.map(toLine).join("\n") + "\n");
writeFileSync("finetune/eval.jsonl", evalSet.map(toLine).join("\n") + "\n");
writeFileSync(
  "finetune/regression.jsonl",
  REGRESSION.map(toLine).join("\n") + "\n",
);
console.log(
  `train=${train.length} eval=${evalSet.length} regression=${REGRESSION.length}`,
);
