/** FIFO task queue: enqueued async jobs run strictly in submission order.
 * A job's rejection propagates to its own caller but never blocks the next
 * job. Used by the command bar so typing doesn't wait on inference
 * (docs/prd/command-queue.md). */
export function createQueue() {
  let chain: Promise<void> = Promise.resolve();
  return (job: () => Promise<void>): Promise<void> => {
    const run = chain.then(job);
    // Only the chain link swallows — callers still see their own rejection.
    chain = run.catch(() => {});
    return run;
  };
}
