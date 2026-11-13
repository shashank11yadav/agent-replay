import { useEffect, useState } from "react";
import { parseRoute, type Route } from "./router";
import { DiffPage } from "./pages/DiffPage";
import { RunPage } from "./pages/RunPage";
import { RunsPage } from "./pages/RunsPage";

function currentRoute(): Route {
  return parseRoute(window.location.hash);
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute);

  useEffect(() => {
    const onChange = () => setRoute(currentRoute());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return (
    <>
      <header>
        <a href="#/">agent-replay</a>
      </header>
      <main>
        {route.page === "runs" && <RunsPage suite={route.suite} />}
        {route.page === "run" && <RunPage id={route.id} step={route.step} />}
        {route.page === "diff" && <DiffPage left={route.left} right={route.right} />}
      </main>
    </>
  );
}
