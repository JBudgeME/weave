import { describe, expect, test } from "bun:test";

import { createQueue } from "./queue";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("createQueue", () => {
  test("jobs run FIFO even when later jobs are faster", async () => {
    const enqueue = createQueue();
    const order: number[] = [];
    const a = enqueue(async () => {
      await sleep(30);
      order.push(1);
    });
    const b = enqueue(async () => {
      order.push(2);
    });
    await Promise.all([a, b]);
    expect(order).toEqual([1, 2]);
  });

  test("a rejected job surfaces to its caller but doesn't block the next", async () => {
    const enqueue = createQueue();
    const order: string[] = [];
    let err: Error | undefined;
    await enqueue(async () => {
      throw new Error("boom");
    }).catch((e: Error) => (err = e));
    expect(err?.message).toBe("boom");
    await enqueue(async () => {
      order.push("after");
    });
    expect(order).toEqual(["after"]);
  });
});
