/**
 * The tool catalog — the single vocabulary shared by the model (Needle or
 * mock), the API layer, and the templates. Needle-style schema.
 */

export const TOOLS = [
  {
    name: "show_tasks",
    description: "Display the user's task list.",
    parameters: {
      filter: {
        type: "string",
        description: "Which tasks to show: all, open, or done.",
        required: false,
      },
      due: {
        type: "string",
        description: "Only show tasks due then, e.g. Fri or tomorrow.",
        required: false,
      },
    },
  },
  {
    name: "add_task",
    description: "Add a new task to the user's task list.",
    parameters: {
      title: {
        type: "string",
        description: "What the task is.",
        required: true,
      },
      due: {
        type: "string",
        description: "When it's due, e.g. Fri or tomorrow.",
        required: false,
      },
    },
  },
  {
    name: "complete_task",
    description: "Mark a task on the user's task list as done.",
    parameters: {
      title: {
        type: "string",
        description: "The task to mark done (partial match ok).",
        required: true,
      },
    },
  },
  {
    name: "edit_task",
    description: "Change a task's title and/or due date.",
    parameters: {
      title: {
        type: "string",
        description: "The task to change (partial match ok).",
        required: true,
      },
      new_title: {
        type: "string",
        description: "The new title, if renaming.",
        required: false,
      },
      new_due: {
        type: "string",
        description: "The new due date, if rescheduling.",
        required: false,
      },
    },
  },
  {
    name: "uncomplete_task",
    description: "Mark a done task as not done again.",
    parameters: {
      title: {
        type: "string",
        description: "The task to reopen (partial match ok).",
        required: true,
      },
    },
  },
  {
    name: "delete_task",
    description: "Remove a task from the user's task list entirely.",
    parameters: {
      title: {
        type: "string",
        description: "The task to remove (partial match ok).",
        required: true,
      },
    },
  },
] as const;
