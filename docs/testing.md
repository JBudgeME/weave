# Testing

Tests run on **`bun:test`** (built-in). Files match `*.test.ts` / `*.test.tsx`, colocated with the code they test.

DOM tests use `@testing-library/react` over `happy-dom`. Preload setup lives in `test/happy-dom.ts` (registers happy-dom) and `test/setup.ts` (extends `expect` with `jest-dom` matchers, runs `cleanup` after each test). The two preloads are wired in order via `bunfig.toml` — keep them separate; happy-dom must register before any `@testing-library/*` module evaluates, or `document.body` is undefined and `screen` becomes a no-op.

Write tests using `bun:test`'s `describe` / `it` / `expect` API (Jest-compatible). For component tests, prefer `userEvent.setup()` over `fireEvent` for accurate user interaction simulation.

Sample tests at `lib/utils.test.ts` and `components/ui/button.test.tsx` show the patterns.
