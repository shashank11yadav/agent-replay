import { useEffect } from "react";

/** Set the browser tab title for the current page. */
export function useTitle(title: string) {
  useEffect(() => {
    document.title = `${title} · agent-replay`;
  }, [title]);
}
