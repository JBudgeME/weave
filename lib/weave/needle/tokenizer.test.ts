import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { NeedleTokenizer, type TokenizerData } from "./tokenizer";

const ROOT = join(import.meta.dir, "..", "..", "..");
const TOK_PATH = join(ROOT, "public", "needle", "tokenizer.json");
const FIX = join(import.meta.dir, "__fixtures__");

const hasTok = existsSync(TOK_PATH);
const load = <T>(p: string): T => JSON.parse(readFileSync(p, "utf-8")) as T;

describe.if(hasTok)("NeedleTokenizer parity", () => {
  const tok = new NeedleTokenizer(load<TokenizerData>(TOK_PATH));

  test("encode: all fixture (query,tools) match Python token ids", () => {
    const cases = load<
      Array<{ query: string; tools: string; enc_tokens: number[] }>
    >(join(FIX, "encode.json"));
    // enc_tokens = encode(query) + [tools_sep] + encode(tools); check both parts.
    const TOOLS_SEP = 5;
    let checked = 0;
    for (const c of cases) {
      const sep = c.enc_tokens.indexOf(TOOLS_SEP);
      const qExpected = c.enc_tokens.slice(0, sep);
      const tExpected = c.enc_tokens.slice(sep + 1);
      expect(tok.encode(c.query)).toEqual(qExpected);
      // tools may be truncated in the fixture; compare against the same length.
      expect(tok.encode(c.tools).slice(0, tExpected.length)).toEqual(tExpected);
      checked++;
    }
    expect(checked).toBeGreaterThan(250);
  });

  test("decode: fixture id sequences match Python text", () => {
    const cases = load<Array<{ ids: number[]; text: string }>>(
      join(FIX, "decode.json"),
    );
    expect(cases.length).toBeGreaterThan(10);
    for (const c of cases) expect(tok.decode(c.ids)).toBe(c.text);
  });
});

test.skipIf(hasTok)(
  "tokenizer fixtures require public/needle/tokenizer.json (run scripts/export-weights.py)",
  () => {},
);
