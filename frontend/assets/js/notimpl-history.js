/**
 * Analysis history — honest empty state.
 *
 * This page previously rendered fabricated past scans. There is no
 * persistence layer in the backend, so there is nothing real to show. Rather than
 * display invented data, it states the limitation.
 */

import { renderStatus } from "./render.js";

function init() {
  const container = document.getElementById("cs-results");
  if (!container) return;

  renderStatus(container, {
    icon: "history",
    title: "Analysis history is not implemented",
    message:
      "Storing past analyses requires a database and a GET /analysis/{id} endpoint, neither of which exists yet. Results from the current browser session are visible on the Analysis Results page.",
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
