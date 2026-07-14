"use client";

import { useEffect } from "react";

import { Button } from "@/components/ui/button";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
      <h1 className="text-2xl font-semibold tracking-tight">
        Something went wrong.
      </h1>
      <p className="text-muted-foreground text-sm">
        An unexpected error occurred. Try again, or refresh the page.
      </p>
      <Button onClick={() => reset()}>Try again</Button>
    </main>
  );
}
