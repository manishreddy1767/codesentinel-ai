/**
 * Report generation — honest empty state.
 *
 * This page previously rendered fabricated report summaries. There is no
 * stored analysis history in the backend, so there is nothing real to show. Rather than
 * display invented data, it states the limitation.
 */

import { renderStatus } from "./render.js";

function init() {
  const container = document.getElementById("cs-results");
  if (!container) return;

  renderStatus(container, {
    icon: "summarize",
    title: "Report generation is not implemented",
    message:
      "Reports aggregate stored analyses, which requires the persistence layer that is not implemented. The Analysis Results page shows the current session's run.",
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
