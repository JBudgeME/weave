/**
 * Client-side task store, localStorage-backed so it survives reloads.
 * Falls back to defaults on the server (all callers are client components).
 */

export interface StoredTask {
  title: string;
  status: "open" | "done";
  due: string;
}

export interface StoreShape {
  tasks: StoredTask[];
}

const KEY = "weave-store-v1";

const DEFAULTS: StoreShape = {
  tasks: [
    { title: "Finetune Needle on the tool catalog", status: "open", due: "—" },
    { title: "Port Needle to WASM", status: "open", due: "—" },
    { title: "Strip greenfield branding", status: "done", due: "Mon" },
  ],
};

function load(): StoreShape {
  if (typeof window === "undefined") return structuredClone(DEFAULTS);
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return structuredClone(DEFAULTS);
    const parsed = JSON.parse(raw) as Partial<StoreShape>;
    return { tasks: parsed.tasks ?? structuredClone(DEFAULTS).tasks };
  } catch {
    return structuredClone(DEFAULTS);
  }
}

export function getStore(): StoreShape {
  return load();
}

export function updateStore(fn: (s: StoreShape) => void): StoreShape {
  const s = load();
  fn(s);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(KEY, JSON.stringify(s));
  }
  return s;
}
