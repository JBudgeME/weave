/**
 * Pure-TS SentencePiece BPE tokenizer, byte-fallback, identity normalizer.
 * Mirrors the Python NeedleTokenizer (sentencepiece model_type=BPE).
 *
 * Encoding: normalize -> split into codepoints -> greedy BPE merges by piece
 * score (max-heap, ties broken by leftmost position) -> emit ids, falling back
 * to <0xNN> byte pieces for symbols absent from the vocab.
 */

const SPACE = "▁"; // ▁ meta-space

export interface TokenizerData {
  pieces: string[];
  scores: number[];
  types: number[]; // 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 6 BYTE
  unk_id: number;
  normalizer: {
    add_dummy_prefix: boolean;
    remove_extra_whitespaces: boolean;
    escape_whitespaces: boolean;
  };
  byte_fallback: boolean;
}

export class NeedleTokenizer {
  private pieceToId = new Map<string, number>();
  private scores: number[];
  private types: number[];
  private pieces: string[];
  private unkId: number;
  private byteToId = new Int32Array(256).fill(-1);
  private norm: TokenizerData["normalizer"];
  private byteFallback: boolean;

  constructor(data: TokenizerData) {
    this.pieces = data.pieces;
    this.scores = data.scores;
    this.types = data.types;
    this.unkId = data.unk_id;
    this.norm = data.normalizer;
    this.byteFallback = data.byte_fallback;
    for (let i = 0; i < data.pieces.length; i++) {
      const p = data.pieces[i];
      // First occurrence wins (matches sentencepiece PieceToId).
      if (!this.pieceToId.has(p)) this.pieceToId.set(p, i);
      if (data.types[i] === 6) {
        // BYTE piece "<0xNN>"
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(p);
        if (m) this.byteToId[parseInt(m[1], 16)] = i;
      }
    }
  }

  private normalize(text: string): string {
    let s = text;
    if (this.norm.remove_extra_whitespaces) {
      s = s.replace(/ +/g, " ").replace(/^ | $/g, "");
    }
    if (this.norm.add_dummy_prefix) s = " " + s;
    if (this.norm.escape_whitespaces) s = s.replace(/ /g, SPACE);
    return s;
  }

  encode(text: string): number[] {
    const norm = this.normalize(text);
    if (norm.length === 0) return [];
    const symbols = Array.from(norm); // codepoints

    // Doubly-linked list over symbol slots.
    const n = symbols.length;
    const piece = symbols.slice();
    const prev = new Int32Array(n);
    const next = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      prev[i] = i - 1;
      next[i] = i === n - 1 ? -1 : i + 1;
    }

    type Pair = { left: number; right: number; score: number; size: number };
    // Max-heap by score; tie -> smaller left index first.
    const heap: Pair[] = [];
    const less = (a: Pair, b: Pair) =>
      a.score < b.score || (a.score === b.score && a.left > b.left);
    const push = (p: Pair) => {
      heap.push(p);
      let i = heap.length - 1;
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (less(heap[parent], heap[i])) {
          [heap[parent], heap[i]] = [heap[i], heap[parent]];
          i = parent;
        } else break;
      }
    };
    const pop = (): Pair => {
      const top = heap[0];
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        let i = 0;
        const len = heap.length;
        for (;;) {
          const l = 2 * i + 1;
          const r = 2 * i + 2;
          let big = i;
          if (l < len && less(heap[big], heap[l])) big = l;
          if (r < len && less(heap[big], heap[r])) big = r;
          if (big === i) break;
          [heap[big], heap[i]] = [heap[i], heap[big]];
          i = big;
        }
      }
      return top;
    };

    const tryAdd = (left: number, right: number) => {
      if (left === -1 || right === -1) return;
      const merged = piece[left] + piece[right];
      const id = this.pieceToId.get(merged);
      if (id === undefined) return;
      push({ left, right, score: this.scores[id], size: merged.length });
    };

    for (let i = 0; i + 1 < n; i++) tryAdd(i, i + 1);

    while (heap.length > 0) {
      const top = pop();
      const { left, right } = top;
      // Stale if either slot consumed or the pair no longer adjacent/sized.
      if (piece[left] === "" || piece[right] === "") continue;
      if (piece[left].length + piece[right].length !== top.size) continue;
      if (next[left] !== right) continue;

      piece[left] = piece[left] + piece[right];
      piece[right] = "";
      const nr = next[right];
      next[left] = nr;
      if (nr !== -1) prev[nr] = left;

      tryAdd(prev[left], left);
      tryAdd(left, next[left]);
    }

    const out: number[] = [];
    for (let i = 0; i !== -1; i = next[i]) {
      const p = piece[i];
      if (p === "") continue;
      const id = this.pieceToId.get(p);
      if (id !== undefined) {
        out.push(id);
      } else if (this.byteFallback) {
        const bytes = new TextEncoder().encode(p);
        for (const b of bytes) {
          const bid = this.byteToId[b];
          out.push(bid !== -1 ? bid : this.unkId);
        }
      } else {
        out.push(this.unkId);
      }
    }
    return out;
  }

  /** Decode ids to text, mirroring sentencepiece DecodePieces. */
  decode(ids: number[]): string {
    // Assemble, reconstructing raw bytes from byte pieces before UTF-8 decode.
    const bytes: number[] = [];
    const pushStr = (s: string) => {
      for (const b of new TextEncoder().encode(s)) bytes.push(b);
    };
    for (const id of ids) {
      if (id < 0 || id >= this.pieces.length) continue;
      const t = this.types[id];
      const p = this.pieces[id];
      if (t === 6) {
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(p);
        if (m) bytes.push(parseInt(m[1], 16));
      } else if (t === 3) {
        // control (pad/eos/bos) -> emit nothing
      } else {
        pushStr(p);
      }
    }
    let text = new TextDecoder().decode(new Uint8Array(bytes));
    text = text.replace(new RegExp(SPACE, "g"), " ");
    // Remove single leading space from add_dummy_prefix.
    if (this.norm.add_dummy_prefix && text.startsWith(" "))
      text = text.slice(1);
    return text;
  }
}
