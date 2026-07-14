import { WeaveTodo } from "@/components/weave-todo";

const GITHUB_URL = "https://github.com/JBudgeME/weave";

function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="currentColor"
      className={className}
      aria-hidden
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

const BADGES = ["MIT", "WebGPU", "26M params", "no server"];

const PIPELINE = [
  "plain English",
  "26M-param model",
  "tool call",
  "template",
  "UI",
];

const NUMBERS = [
  { value: "26M", label: "parameters — small enough to ship to a browser tab" },
  {
    value: "100%",
    label:
      "accuracy on our own 442-example held-out eval set (six tools, generated data)",
  },
  {
    value: "~350ms",
    label:
      "median command latency with WebGPU on a desktop GPU; a few seconds on the TS fallback",
  },
  { value: "52MB", label: "one-time model download, then browser-cached" },
  { value: "0", label: "servers — the deployed site is static files" },
  { value: "~3min", label: "to retrain on new tools with a consumer GPU" },
];

const FEATURES = [
  {
    title: "Constrained output",
    body: "The model never writes UI or JSON structure — it picks a tool from a fixed catalog and fills its args. Hand-written templates render the result, which keeps the UI well-formed even when the model is wrong.",
  },
  {
    title: "Three backends",
    body: "WebGPU compute shaders when your browser supports them, a pure-TypeScript fallback when it doesn't, and a regex mock if the download fails. Every answer is tagged with the backend that produced it.",
  },
  {
    title: "Runs locally",
    body: "There's no inference server — commands are processed in a Web Worker in your tab. The deployed site is static HTML, JS, and one weight file, so your input stays on your machine.",
  },
];

const linkCls =
  "underline underline-offset-4 transition-colors hover:text-foreground";

export default function Home() {
  return (
    <div className="flex w-full flex-col">
      {/* Hero */}
      <section className="mx-auto w-full max-w-3xl px-4 pt-12 pb-8 text-center">
        <p className="mb-3">
          <span className="rounded-full border border-violet-500/40 bg-violet-500/10 px-3 py-1 text-xs font-medium text-violet-700 dark:text-violet-300">
            proof of concept
          </span>
        </p>
        <h1 className="bg-gradient-to-r from-violet-500 via-fuchsia-500 to-violet-400 bg-clip-text text-4xl font-bold tracking-tight text-transparent sm:text-5xl">
          weave
        </h1>
        <p className="text-muted-foreground mx-auto mt-4 max-w-xl text-lg leading-relaxed">
          A todo app driven by typed plain-English commands — powered by a
          26M-parameter model running entirely in your browser.
        </p>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
          {BADGES.map((b) => (
            <span
              key={b}
              className="border-border text-muted-foreground rounded-full border px-2.5 py-0.5 font-mono text-xs"
            >
              {b}
            </span>
          ))}
        </div>
        <div className="mt-6 flex items-center justify-center gap-3">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex cursor-pointer items-center gap-2 rounded-full bg-violet-600 px-5 py-2.5 text-sm font-medium text-white shadow transition-colors duration-200 hover:bg-violet-500"
          >
            <GitHubIcon className="size-4" /> View on GitHub
          </a>
          <a
            href="#how-it-works"
            className="border-border hover:bg-accent inline-flex cursor-pointer items-center rounded-full border px-5 py-2.5 text-sm transition-colors duration-200"
          >
            How it works
          </a>
        </div>
      </section>

      {/* Live demo */}
      <section className="mx-auto flex w-full max-w-xl flex-1 flex-col px-4">
        <p className="text-muted-foreground pb-2 text-center font-mono text-xs">
          live demo — the model loads on your first command (52MB, once)
        </p>
        <div className="border-border bg-card flex min-h-0 flex-col rounded-2xl border p-4 shadow-sm">
          <WeaveTodo />
        </div>
      </section>

      {/* What is this */}
      <section className="mx-auto w-full max-w-3xl px-4 py-16">
        <h2 className="text-2xl font-semibold tracking-tight">What is this?</h2>
        <p className="text-muted-foreground mt-4 leading-relaxed">
          A proof of concept for <strong>tiny-model generative UI</strong>. You
          type plain English; a small neural network — downloaded once and run
          locally in your browser — maps it to one of six todo actions. The
          point isn&apos;t the todo list: it&apos;s that a 26M-parameter model,
          finetuned in minutes on a few thousand generated examples, handles
          this kind of narrow natural-language → tool-call routing reliably
          (100% on our held-out eval set). For a small, well-defined command
          surface, you may not need a frontier model — or a server.
        </p>
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="border-border rounded-2xl border p-5 transition-colors duration-200 hover:border-violet-500/50"
            >
              <h3 className="font-medium">{f.title}</h3>
              <p className="text-muted-foreground mt-2 text-sm leading-relaxed">
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="border-border/40 border-t">
        <div className="mx-auto w-full max-w-3xl px-4 py-16">
          <h2 className="text-2xl font-semibold tracking-tight">
            How it works
          </h2>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-2 font-mono text-sm">
            {PIPELINE.map((step, i) => (
              <span key={step} className="flex items-center gap-2">
                <span
                  className={
                    step === "26M-param model"
                      ? "rounded-lg border border-violet-500/60 bg-violet-500/10 px-3 py-1.5 text-violet-700 dark:text-violet-300"
                      : "border-border bg-card rounded-lg border px-3 py-1.5"
                  }
                >
                  {step}
                </span>
                {i < PIPELINE.length - 1 && (
                  <span className="text-muted-foreground" aria-hidden>
                    →
                  </span>
                )}
              </span>
            ))}
          </div>
          <p className="text-muted-foreground mt-8 leading-relaxed">
            <a
              className={linkCls}
              href="https://github.com/cactus-compute/needle"
              target="_blank"
              rel="noreferrer"
            >
              Needle
            </a>{" "}
            (an encoder-decoder transformer distilled from Gemini) was finetuned
            on ~3,900 deterministic slot-fill examples, then hand-ported to
            dependency-free TypeScript. The encoder runs in WebGPU compute
            shaders when available; the KV-cached decoder runs in TypeScript;
            everything lives in a Web Worker so the page never freezes.
            Rendering goes through{" "}
            <a
              className={linkCls}
              href="https://github.com/vercel-labs/json-render"
              target="_blank"
              rel="noreferrer"
            >
              json-render
            </a>{" "}
            templates. A parity harness holds every backend to 100% routing /
            100% exact-args against the Python reference —{" "}
            <a className={linkCls} href="/dev/parity">
              run it yourself
            </a>
            .
          </p>
        </div>
      </section>

      {/* By the numbers */}
      <section className="border-border/40 border-t">
        <div className="mx-auto w-full max-w-3xl px-4 py-16">
          <h2 className="text-2xl font-semibold tracking-tight">
            By the numbers
          </h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {NUMBERS.map((n) => (
              <div
                key={n.value}
                className="border-border rounded-2xl border p-5"
              >
                <div className="text-3xl font-bold tracking-tight text-violet-600 tabular-nums dark:text-violet-400">
                  {n.value}
                </div>
                <p className="text-muted-foreground mt-1.5 text-sm leading-relaxed">
                  {n.label}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick start */}
      <section className="border-border/40 border-t">
        <div className="mx-auto w-full max-w-3xl px-4 py-16">
          <h2 className="text-2xl font-semibold tracking-tight">
            Run it yourself
          </h2>
          <pre className="bg-card border-border mt-6 overflow-x-auto rounded-2xl border p-5 font-mono text-sm leading-relaxed">
            {`git clone ${GITHUB_URL}.git
cd weave
bun install
bun dev   # http://localhost:3000`}
          </pre>
          <p className="text-muted-foreground mt-4 text-sm">
            Training your own checkpoint (and the full story) is in the{" "}
            <a
              className={linkCls}
              href={`${GITHUB_URL}#readme`}
              target="_blank"
              rel="noreferrer"
            >
              README
            </a>
            .
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-border/40 border-t">
        <div className="text-muted-foreground mx-auto flex w-full max-w-3xl flex-wrap items-center justify-between gap-2 px-4 py-8 text-xs">
          <span>MIT © 2026 Jason Budge</span>
          <span>
            built with{" "}
            <a
              className={linkCls}
              href="https://github.com/cactus-compute/needle"
              target="_blank"
              rel="noreferrer"
            >
              Needle
            </a>
            {" · "}
            <a
              className={linkCls}
              href="https://github.com/vercel-labs/json-render"
              target="_blank"
              rel="noreferrer"
            >
              json-render
            </a>
            {" · "}
            <a
              className={linkCls}
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
            >
              source
            </a>
          </span>
        </div>
      </footer>
    </div>
  );
}
